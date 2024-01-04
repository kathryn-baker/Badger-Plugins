import logging
import multiprocessing as mp
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
from badger import interface
from joblib import Parallel, delayed
from p4p.client.thread import Context


class Interface(interface.Interface):
    name = "epics_pva"
    """Concrete interface for interacting with EPICS PVAccess PVs"""

    def __init__(self, poll_period=0.1, timeout=3, parallel=False):
        self.poll_period = poll_period
        self.timeout = timeout
        self.parallel = parallel
        super().__init__()

    def get_default_params(self) -> dict:
        return {"context": "pva"}

    def get_value(self, channel: str):
        context = Context("pva")
        try:
            return context.get(channel).raw.value
        except TimeoutError as e:
            # TODO - decide whether we should return a NaN value here
            logging.exception(f"{channel}: {e}")
            return np.nan
            # raise e
        finally:
            context.close()

    def get_values(self, channels: List[str]) -> Dict[str, float]:
        time.sleep(self.poll_period)
        context = Context("pva")
        try:
            values = [value.raw.value for value in context.get(channels)]
        except TimeoutError:
            logging.exception(f"Timeout error from {channels}, retrying individually")
            # if we get a timeout error one even one of them, we retry to get
            # the individual values
            values = [self.get_value(channel) for channel in channels]
        context.close()
        return dict(zip(channels, values))

    def set_value(
        self,
        channel: str,
        value,
        set_config=None,
        # validate_readback=False,
        # readback_pv=None,
        # tolerance=1e-3,
        # count_down=10,
        # offset=0,
    ):
        # for parallel to work, context has to be made and closed within the function
        context = Context("pva")
        start_time = time.time()
        # always put the value to the set PV
        try:
            context.put(channel, value, timeout=self.timeout, get=True)
        except TypeError:
            context.put(channel, value.item(), timeout=self.timeout, get=True)
        logging.debug(f"put value {value} to {channel}")
        # then, if configured to look at a readback PV, do the check on the PV
        if set_config is not None and set_config.get("validate_readback", False):
            readback_pv = set_config.get("readback_pv")
            tolerance = set_config.get("tolerance", 1e-3)
            count_down = set_config.get("count_down", 10)
            time_limit = deepcopy(count_down)
            offset = set_config.get("offset", 0)
            while count_down > 0:
                # replace with monitor and conditional variables
                _value = context.get(readback_pv, timeout=self.timeout).raw.value
                if np.isclose(_value, value + offset, rtol=tolerance):
                    # should we return the set value or the read value here?
                    end_time = time.time()
                    logging.debug(
                        f"Set var for {channel} took {end_time - start_time:5.5f}s"
                    )
                    context.close()
                    return _value

                time.sleep(0.1)
                count_down -= 0.1
            logging.exception(
                f"PV {channel} (current: {_value}) cannot reach expected value ({value}) in designated time {time_limit}!"
            )
        else:
            end_time = time.time()
            logging.debug(f"Set var for {channel} took {end_time - start_time:5.5f}s")
        context.close()
        return value

    def set_values(self, channels, values, configs: Dict[str, dict]):
        start = time.time()
        if self.parallel:
            Parallel(n_jobs=mp.cpu_count())(
                delayed(self.set_value)(channel, value, configs.get(channel))
                for channel, value in zip(channels, values)
            )
        else:
            for channel, value in zip(channels, values):
                # TODO - what should we do if we can't get to the right value?
                self.set_value(channel, value, configs.get(channel))
        end = time.time()
        logging.info(f"total set time: {end - start:.5f}s")
