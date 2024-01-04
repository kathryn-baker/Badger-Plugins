import json
import logging
from typing import List

from paho.mqtt.client import Client


class MQTTClient(Client):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.messages = {}
        super().__init__(*args, **kwargs)

    def on_connect(self, client, userdata, flags, rc):
        logging.info(f"Connected with result code:{str(rc)}")

    def on_disconnect(self, client, userdata, rc):
        logging.info(f"Disconnected with result code:{str(rc)}")

    def on_publish(self, client, userdata, mid):
        logging.debug(f"Message published! ID: {mid}")

    def on_log(self, client, userdata, level, buf):
        if level > 20:
            # we only want to log messages over the INFO level
            logging.info(level=level, msg=buf)

    def on_message(self, client, userdata, message):
        message_byte = message.payload
        message_dict = json.loads(message_byte.decode("utf-8"))
        channel_name = message_dict["channel"]

        if channel_name.upper() in list(self.messages.keys()):
            self.messages[channel_name.upper()].append(message_dict["value"])
        else:
            self.messages[channel_name.upper()] = [message_dict["value"]]

    def dump_messages(self):
        self.messages = {}

    def initialise_messages(self, channels: List[str]):
        self.messages = {channel.upper(): [] for channel in channels}

    def subscribe(self, topic, qos=0, options=None, properties=None) -> tuple[int, int]:
        logging.debug(f"subscribed to {topic}")
        return super().subscribe(topic, qos, options, properties)


class ValidationClient(Client):
    def __init__(self, monitor_topic, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.monitor_topic = monitor_topic
        self.current_value = None

    def on_connect(self, client, userdata, flags, rc):
        self.subscribe(self.monitor_topic)
        logging.info(
            f"\nValidation client connected with result code:{str(rc)}, subscribed to {self.monitor_topic}"
        )

    def on_disconnect(self, client, userdata, rc):
        self.unsubscribe(self.monitor_topic)
        logging.info(f"\nValidation client disconnected with result code:{str(rc)}")

    def on_message(self, client, userdata, message):
        logging.debug("message received!")
        message_byte = message.payload
        message_dict = json.loads(message_byte.decode("utf-8"))
        self.current_value = message_dict["value"]
