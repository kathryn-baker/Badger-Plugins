import json
import logging
import multiprocessing as mp
import random
import string
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
from badger import interface
from joblib import Parallel, delayed

from .clients import MQTTClient, ValidationClient


def pv_name_to_mqtt_topic(pvname: str, mode="get"):
    topic = pvname.replace("::", ":").replace(":", "/").lower()
    if mode == "get":
        prefix = "values"
        suffix = ""
    elif mode == "set":
        prefix = "set"
        suffix = "/value"
    return f"vista/{prefix}/{topic}{suffix}"


def generate_shortuuid() -> str:
    """Public function for generating short UUID messages to be attached
    to MQTT messages as message_id's"""
    alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits
    shortuiid = "".join(random.choices(alphabet, k=8))
    return shortuiid


class Interface(interface.Interface):
    name = "mqtt"
    """Concrete interface for interacting with Vsystem via MQTT messages"""

    def __init__(
        self,
        host="mosquitto",
        port=1883,
        keepalive=60,
        poll_period=0.1,
        timeout=3,
    ):
        self.host = host
        self.port = port
        self.keepalive = keepalive
        self.poll_period = poll_period
        self.timeout = timeout
        super().__init__()

    def get_default_params(self) -> dict:
        params = {
            "url": self.host,
            "port": self.port,
            "keepalive": self.keepalive,
        }
        return params

    def set_value(
        self,
        channel: str,
        value,
        validate_readback=False,
        readback_pv=None,
        tolerance=1e-3,
        count_down=10,
        offset=0,
    ):
        client = MQTTClient()
        start_time = time.time()
        time_limit = deepcopy(count_down)
        # always put the value to the set PV
        set_topic = pv_name_to_mqtt_topic(channel, mode="set")
        if isinstance(value, np.ndarray):
            value = value.item()

        payload = {
            "timestamp": time.time(),
            "channel": channel.lower(),
            "value": value,
            "messageid": generate_shortuuid(),
        }
        client.connect(self.host, self.port, self.keepalive)
        client.publish(topic=set_topic, payload=json.dumps(payload))
        logging.debug(f"Published value {value} to {set_topic}")
        # then, if configured to look at a readback PV, do the check on the PV
        if validate_readback:
            readback_topic = pv_name_to_mqtt_topic(readback_pv, mode="get")
            validation_client = ValidationClient(
                monitor_topic=readback_topic,
            )
            validation_client.connect(self.host, self.port, self.keepalive)
            validation_client.loop_start()
            set_correct = False
            while count_down > 0:
                # set_correct = False
                _value = validation_client.current_value
                if _value is not None:
                    if np.isclose(_value, value + offset, rtol=tolerance):
                        # should we return the set value or the read value here?
                        end_time = time.time()
                        logging.info(
                            f"Set var for {channel} took {end_time - start_time:5.5f}s"
                        )
                        set_correct = True
                        break

                time.sleep(0.1)
                count_down -= 0.1
            # always stop the client once the validation is complete
            validation_client.loop_stop()
            validation_client.disconnect()
            client.loop_stop()
            client.disconnect()
            if set_correct is False:
                raise Exception(
                    f"PV {channel} (current: {_value}) cannot reach expected value ({value}) in designated time {time_limit}!"
                )
            else:
                return _value
        else:
            client.loop_stop()
            client.disconnect()
            end_time = time.time()
            logging.info(f"Set var for {channel} took {end_time - start_time:5.5f}s")
            return value

    def set_values(self, channels, values, configs: Dict[str, dict], parallel=False):
        start = time.time()
        if parallel:
            Parallel(n_jobs=mp.cpu_count())(
                delayed(self.set_value)(channel, value, **configs[channel])
                for channel, value in zip(channels, values)
            )
        else:
            for channel, value in zip(channels, values):
                config = configs[channel]
                self.set_value(channel, value, **config)
        end = time.time()
        logging.info(f"total set time: {end - start:.5f}s")

    def get_value(self, channel: str):
        # we don't use this here because it's a stream of data instead of a request
        return self.get_values([channel])[channel]

    def get_values(self, channels: List[str]) -> Dict[str, float]:
        # in order to get values from MQTT, we need to start the loop and build
        # up a message bank that we can then parse afterwards
        client = MQTTClient(self.host, self.port, self.keepalive)

        client.initialise_messages(channels)
        self._read_system(client, channels)
        results = {}
        for channel, value in client.messages.items():
            if len(value) == 0:
                results[channel] = np.nan
            else:
                results[channel] = value[-1]
        client.dump_messages()

        return results

    def _read_system(self, client, channels: List[str]):
        client.connect(self.host, self.port, self.keepalive)
        topics = [pv_name_to_mqtt_topic(channel, mode="get") for channel in channels]
        client.subscribe([(topic, 0) for topic in topics])
        client.loop_start()
        time.sleep(self.poll_period)
        client.loop_stop()
        client.unsubscribe(topics)
        client.disconnect()
