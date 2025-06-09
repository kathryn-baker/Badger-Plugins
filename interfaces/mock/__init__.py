from badger import interface

from numpy import random


class Interface(interface.Interface):
    name = "mock"

    @staticmethod
    def get_default_params():
        return None

    def get_value(self, channel: str):
        print("Called get_value for channel: {}.".format(channel))
        return random.random()

    def set_value(self, channel: str, value):
        print("Called set_value for channel: {}, with value: {}".format(channel, value))

    def get_values(self, channels: list):
        return {channel: self.get_value(channel) for channel in channels}

    def set_values(self, channel_inputs: dict, *args, **kwargs):
        print(f"Called set_values with args:{args}, kwargs:{kwargs}")
        for channel, val in channel_inputs.items():
            self.set_value(channel, val)
