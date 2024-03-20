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
            return context.get(channel).raw.value
        except TimeoutError as e:
            # TODO - decide whether we should return a NaN value here
            logging.exception(f"{channel}: {e}")
            return np.nan
            # raise e
        finally:
            context.close()

    def get_values(self, channels: List[str]) -> Dict[str, float]:
        # time.sleep(self.poll_period)
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
        context.close()
        return dict(zip(channels, values))
    def _put(self, context, channel, value):
        try:
            context.put(channel, value, timeout=self.timeout, get=True)
        except TypeError:
            context.put(channel, value.item(), timeout=self.timeout, get=True)
        except TimeoutError as e:
            raise
        except Exception as e:
            print(f'Failed to put value {value} to {channel} because of {e}')

    def set_value(
        self,
        channel: str,
        value,
        set_config=None,
    ):
        if not self.read_only:
            # for parallel to work, context has to be made and closed within the function
            context = Context("pva")
            start_time = time.time()
            # always put the value to the set PV
            try:
                self._put(context, channel, value)
            #     context.put(channel, value, timeout=self.timeout, get=True)
            # except TypeError:
            #     context.put(channel, value.item(), timeout=self.timeout, get=True)
            except TimeoutError:
                logging.exception(f'Timeout Error on {channel}')
                # we try again!
                try:
                    time.sleep(1)
                    self._put(context, channel, value)
                except TimeoutError as e:
                    # give it some time and see if it fixes itself
                    logging.exception(f'Timeout on {channel}: {e} after 2 attempts and a 1 second delay')
                    # raise TimeoutError(f'Timeout on {channel}: {e} after 2 tries and ')
                # raise e
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
                    _value = context.get(readback_pv, timeout=self.timeout)
                    severity = _value.severity
                    _value = _value.real
                    if severity != 3:
                        if np.isclose(_value, value + offset, atol=tolerance):
                            # should we return the set value or the read value here?
                            end_time = time.time()
                            logging.debug(
                                f"Set var for {channel} took {end_time - start_time:5.5f}s"
                            )
                            context.close()
                            return _value

                        time.sleep(0.1)
                        count_down -= 0.1
                    else:
                        logging.warning(f'readback PV {readback_pv} is in an I/O error state, validation will continue once it is out of this state')
                        time.sleep(2)
                logging.exception(
                    f"PV {channel} (current: {_value}) cannot reach expected value ({value}) in designated time {time_limit}!"
                )
            else:
                end_time = time.time()
                logging.debug(
                    f"Set var for {channel} took {end_time - start_time:5.5f}s"
                )
            context.close()
            return value
        else:
            logging.info(
                f"Interface is set to read-only mode, cannot set value {value} to {channel}"
            )

    def set_values(self, channels, values, configs: Dict[str, dict]):
        if not self.read_only:
            start = time.time()
            if self.parallel:
                try:
                    Parallel(n_jobs=mp.cpu_count())(
                        delayed(self.set_value)(channel, value, configs.get(channel))
                        for channel, value in zip(channels, values)
                    )
                except TimeoutError as e:
                    # if we get a TimeoutError, for now we want to just continue with
                    # the update and hope that getting the values after will calrify
                    # what the value actually was
                    # TODO this will have to be dealt with differently for a more 
                    # generalisable solution across facilities as we want to make sure
                    # that the values going into the Xopt model are correct
                    logging.exception(e)
            else:
                for channel, value in zip(channels, values):
                    # TODO - what should we do if we can't get to the right value?
                    self.set_value(channel, value, configs.get(channel))
            end = time.time()
            logging.info(f"total set time: {end - start:.5f}s")
        else:
            logging.info(
                f"Interface is set to read-only mode, cannot set values {values} to {channels}"
            )
