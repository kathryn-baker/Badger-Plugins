import logging
import multiprocessing as mp
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
from badger import interface
from joblib import Parallel, delayed
from p4p.client.thread import Context


def timeit(func):
    def wrapper_timeit(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"total set time: {end_time - start_time:.5f}s")

    return wrapper_timeit


def retry_on_timeout(func):
    def wrapper_retry(*args, **kwargs):
        channel = args[1]
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
    name = "epics_pva"
    """Concrete interface for interacting with EPICS PVAccess PVs"""

    def __init__(self, poll_period=0.1, timeout=3, parallel=False, read_only=False):
        self.poll_period = poll_period
        self.timeout = timeout
        self.parallel = parallel
        self.read_only = read_only
        super().__init__()

    def get_default_params(self) -> dict:
        return {
            "context": "pva",
            "read-only": self.read_only,
        }

    def get_value(self, channel: str):
        context = Context("pva")
        try:
            return context.get(channel).real
        except TimeoutError as e:
            # TODO - decide whether we should return a NaN value here
            logging.exception(f"{channel}: {e}")
            return np.nan
        finally:
            context.close()

    def get_values(self, channels: List[str]) -> Dict[str, float]:
        context = Context("pva")
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
        except TypeError:
            context.put(channel, value.item(), timeout=self.timeout)
        logging.debug(f"put value {value} to {channel}")

    def set_value(self, channel: str, value, validation_function=None):
        if not self.read_only:
            # for parallel to work, context has to be made and closed within the function
            context = Context("pva")
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
                logging.exception(e)
            finally:
                context.close()
        else:
            logging.info(
                "Interface is set to read-only mode, cannot set value %s to %s",
                value,
                channel,
            )

    @timeit
    def set_values(self, channels, values, configs: Dict[str, dict]):
        if not self.read_only:
            if self.parallel:
                Parallel(n_jobs=mp.cpu_count())(
                    delayed(self.set_value)(channel, value, configs.get(channel))
                    for channel, value in zip(channels, values)
                )

            else:
                for channel, value in zip(channels, values):
                    self.set_value(channel, value, configs.get(channel))
        else:
            logging.info(
                f"Interface is set to read-only mode, cannot set values {values} to {channels}"
            )
