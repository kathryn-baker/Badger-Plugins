import logging
import multiprocessing as mp
import time
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List

import numpy as np
from badger import interface
from joblib import Parallel, delayed
from p4p.client.thread import Context
from pydantic import Field


def timeit(func):
    def wrapper_timeit(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"total set time: {end_time - start_time:.5f}s")

    return wrapper_timeit


def retry_on_timeout(func):
    def wrapper_retry(*args, **kwargs):
        channel = args[2]
        try:
            func(*args, **kwargs)
        except TimeoutError:
            # we try again!
            # give it some time and see if it fixes itself
            time.sleep(1)
            try:
                func(*args, **kwargs)
            except TimeoutError as e:
                # TODO - decide whether we should re-raise or not
                raise TimeoutError(
                    f"Timeout on {channel}: {e} after 2 attempts and a 1 second delay"
                )

    return wrapper_retry


class Interface(interface.Interface):
    """Concrete interface for interacting with EPICS PVAccess PVs"""

    name: str = "epics_pva"
    context_str: str = "pva"
    timeout: float = Field(
        default=3.0, description="Number of seconds to try connecting to a PV"
    )
    parallel: bool = Field(
        default=False,
        description="Flag indicating whether all variables should be set at once or in series.",
    )
    read_only: bool = Field(
        default=False,
        description="Flag to indicate whether the interface should allow values to be set or not.",
    )

    def get_default_params(self) -> dict:
        return {
            "context": self.context_str,
            "read-only": self.read_only,
        }

    def get_value(self, channel: str):
        context = Context(self.context_str)
        try:
            return context.get(channel).real
        except TimeoutError as e:
            # TODO - decide whether we should return a NaN value here
            logging.exception(f"{channel}: {e}")
            return np.nan
        finally:
            context.close()

    def get_values(self, channels: List[str]) -> Dict[str, float]:
        if isinstance(channels, str):
            channels = [channels]
        context = Context(self.context_str)
        try:
            # using real allows us to quickly extract the number from both
            # int/floats as well as enum types
            values = [value.real for value in context.get(channels)]
        except TimeoutError:
            logging.exception(f"Timeout error from {channels}, retrying individually")
            # if we get a timeout error one even one of them, we retry to get
            # the individual values
            values = [self.get_value(channel) for channel in channels]
        finally:
            context.close()
        return dict(zip(channels, values))

    @retry_on_timeout
    def _put(self, context, channel, value):
        try:
            context.put(channel, value, timeout=self.timeout)
            logging.debug(f"put value {value} to {channel}")
        except TypeError:
            context.put(channel, value.item(), timeout=self.timeout)

    def set_value(self, channel: str, value, validation_function=None):
        if not self.read_only:
            # for parallel to work, context has to be made and closed within the function
            context = Context(self.context_str)
            # always put the value to the set PV
            try:
                self._put(context, channel, value)
                if validation_function is not None:
                    try:
                        validation_function(
                            set_pv=channel,
                            set_value=value,
                            context=context,
                            timeout=self.timeout,
                        )
                    except ValueError as e:
                        logging.warning(e)
                # context.close()
                # return value
            except TimeoutError as e:
                logging.warning(e)
            finally:
                context.close()
        else:
            logging.info(
                "Interface is set to read-only mode, cannot set value %s to %s",
                value,
                channel,
            )

    @timeit
    def set_values(self, channel_inputs: Dict[str, Any], configs: Dict[str, partial]):
        if not self.read_only:
            if self.parallel:
                Parallel(n_jobs=mp.cpu_count())(
                    delayed(self.set_value)(channel, value, configs.get(channel))
                    for channel, value in channel_inputs.items()
                )

            else:
                for channel, value in channel_inputs.items():
                    self.set_value(channel, value, configs.get(channel))
        else:
            logging.info(
                f"Interface is set to read-only mode, cannot set {channel_inputs}"
            )
