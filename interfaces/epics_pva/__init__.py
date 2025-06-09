import logging
import multiprocessing as mp
import time
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List

import numpy as np
from badger import interface
from joblib import Parallel, delayed
from p4p.client.thread import Cancelled, Context, Disconnected, RemoteError, TimeoutError
from pydantic import Field


def timeit(func):
    def wrapper_timeit(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"total set time: {end_time - start_time:.5f}s")

    return wrapper_timeit


class Interface(interface.Interface):
    """Concrete interface for interacting with EPICS PVAccess PVs"""

    name: str = "epics_pva"
    context_str: str = "pva"
    timeout: float = Field(default=3.0, description="Number of seconds to try connecting to a PV")
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

        result = context.get(channel, throw=False)
        if isinstance(result, (Disconnected, TimeoutError, RemoteError, Cancelled)):
            logging.warning("Could not retrieve value of %s due to %s", channel, type(result).__name__)
            result = np.nan
        else:
            result = result.real
        context.close()
        return result

    def get_values(self, channels: List[str]) -> Dict[str, float]:
        if isinstance(channels, str):
            channels = [channels]
        context = Context(self.context_str)
        book = {}

        results = context.get(channels, throw=False)
        failures = []

        for i, result in enumerate(results):
            if isinstance(result, (Disconnected, TimeoutError, RemoteError, Cancelled)):
                failures.append(channels[i])
                logging.warning(
                    "Could not retrieve value of %s due to %s. Retrying individually.",
                    channels[i],
                    type(result).__name__,
                )
            else:
                book[channels[i]] = result.real

        # then if there are any failures, retry the failed ones in the same way
        if len(failures) >= 1:
            results = context.get(failures, throw=False)

            for i, result in enumerate(results):
                if isinstance(result, (Disconnected, TimeoutError, RemoteError, Cancelled)):
                    logging.warning("Could not retrieve value of %s due to %s.", failures[i], type(result).__name__)
                    book[failures[i]] = np.nan
                else:
                    book[failures[i]] = result.real
        context.close()
        return book

    def _put(self, context, channel, value):
        success = False
        try:
            result = context.put(channel, value, timeout=self.timeout, throw=False)
        except TypeError:
            value = value.item()
            result = context.put(channel, value, timeout=self.timeout, throw=False)
        if result is None:
            logging.debug(
                "Put value %d to %s",
                value,
                channel,
            )
            success = True
        else:
            logging.info("Could not put value %d to %s due to %s. Retrying...", value, channel, type(result).__name__)
            # retry after a short sleep
            time.sleep(1)
            result = context.put(channel, value, timeout=self.timeout, throw=False)
            if result is None:
                logging.debug(
                    "Put value %d to %s",
                    value,
                    channel,
                )
                success = True
            else:
                logging.warning(
                    "Could not put value %d to %s (second attempt) due to %s after a 1 second delay.",
                    value,
                    channel,
                    type(result).__name__,
                )
        return success

    def set_value(self, channel: str, value, validation_function=None):
        if not self.read_only:
            # for parallel to work, context has to be made and closed within the function
            context = Context(self.context_str)
            # always put the value to the set PV

            success = self._put(context, channel, value)
            # if we weren't successful in setting the PV, there's no need to validate
            # against the readback
            if success and validation_function is not None:
                try:
                    validation_function(
                        set_pv=channel,
                        set_value=value,
                        context=context,
                        timeout=self.timeout,
                    )
                except (ValueError, TimeoutError) as e:
                    logging.warning(e)
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
            logging.info("Interface is set to read-only mode, cannot set %s", channel_inputs)
