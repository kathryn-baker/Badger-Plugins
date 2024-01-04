import sys
from pathlib import Path

current_file = Path(__file__)
root_dir = current_file.parents[1]
sys.path.insert(0, str(root_dir))
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

channels = ["test::channel:1", "test::channel:2", "TEST::CHANNEL:3"]
values = [5.0, 6.0, 7.0]

from . import Interface, MQTTClient, ValidationClient


class MockMessage:
    def __init__(self, topic, channel, value) -> None:
        self._topic = topic
        self._payload = {
            "channel": channel,
            "value": value,
            "timestamp": 123.456,
            "messageid": "test1234",
        }

    @property
    def topic(self):
        return self._topic

    @property
    def payload(self):
        return json.dumps(self._payload).encode("utf-8")


def test_mqtt_client_messages():
    client = MQTTClient()

    client.initialise_messages(["test::channel:1", "TEST::CHANNEL:2"])
    assert client.messages == {"TEST::CHANNEL:1": [], "TEST::CHANNEL:2": []}
    client.dump_messages()
    assert client.messages == {}


def test_mqtt_client_on_messages():
    client = MQTTClient()

    test_messages = [
        ("vista/values/test/channel/1", "test::channel:1", 1.0),
        ("vista/values/test/channel/2", "test::channel:2", 2.0),
        ("vista/values/test/channel/1", "test::channel:1", 2.0),
    ]
    for topic, channel, value in test_messages:
        client.on_message(
            client,
            MagicMock(),
            MockMessage(topic, channel, value),
        )

    assert client.messages == {"TEST::CHANNEL:1": [1.0, 2.0], "TEST::CHANNEL:2": [2.0]}


def test_validation_client_on_messages():
    client = ValidationClient("vista/values/test/channel/1")

    test_messages = [
        ("vista/values/test/channel/1", "test::channel:1", 1.0),
        ("vista/values/test/channel/1", "test::channel:2", 2.0),
        ("vista/values/test/channel/1", "test::channel:1", 2.0),
    ]
    for topic, channel, value in test_messages:
        client.on_message(
            client,
            MagicMock(),
            MockMessage(topic, channel, value),
        )
        assert client.current_value == value


@pytest.fixture
def mock_client():
    with patch("mqtt.MQTTClient") as client:
        yield client.return_value


class TestMQTTInterface:
    def test_MQTT_get_values(self, mock_client):
        interface = Interface(
            host="testhost",
            port=1883,
            keepalive=60,
            poll_period=0.1,
            timeout=1,
            parallel=False,
        )

        # the on_message function on the client won't work as a mock because it needs threading
        # so instead we have to fake what the messages might say
        mock_client.messages = {
            "vista/values/test/channel/1": [
                values[0],
                values[0] + 1,
            ],  # an example of multiple values
            "vista/values/test/channel/2": [values[1]],  # an example of just one value
            "vista/values/test/channel/3": [],  # an example where no values are received
        }
        result = interface.get_values(channels)

        # check all the calls
        mock_client.initialise_messages.assert_called_once_with(channels)
        mock_client.loop_start.assert_called_once_with()
        mock_client.loop_stop.assert_called_once_with()
        mock_client.connect.assert_called_once_with("testhost", 1883, 60)
        mock_client.disconnect.assert_called_once_with()
        mock_client.dump_messages.assert_called_once_with()

        mock_client.subscribe.assert_called_once_with(
            [
                ("vista/values/test/channel/1", 0),
                ("vista/values/test/channel/2", 0),
                ("vista/values/test/channel/3", 0),
            ]
        )
        mock_client.unsubscribe.assert_called_once_with(
            [
                "vista/values/test/channel/1",
                "vista/values/test/channel/2",
                "vista/values/test/channel/3",
            ]
        )

        assert result == {
            "vista/values/test/channel/1": values[0] + 1,
            "vista/values/test/channel/2": values[1],
            "vista/values/test/channel/3": np.nan,
        }

    def test_MQTT_get_value(self, mock_client):
        # when we call get_value on it's own, we use the same procedure as with multiple
        # channels but on it's own
        interface = Interface(
            host="testhost",
            port=1883,
            keepalive=60,
            poll_period=0.1,
            timeout=1,
            parallel=False,
        )
        channel_name = "test::channel:1"
        # the on_message function on the client won't work as a mock because it needs threading
        # so instead we have to fake what the messages might say
        mock_client.messages = {
            "vista/values/test/channel/1": [
                values[0],
                values[0] + 1,
            ],  # an example of multiple values
        }
        result = interface.get_value("test::channel:1")

        # check all the calls
        mock_client.initialise_messages.assert_called_once_with([channel_name])
        mock_client.loop_start.assert_called_once_with()
        mock_client.loop_stop.assert_called_once_with()
        mock_client.connect.assert_called_once_with("testhost", 1883, 60)
        mock_client.disconnect.assert_called_once_with()
        mock_client.dump_messages.assert_called_once_with()

        mock_client.subscribe.assert_called_once_with(
            [
                ("vista/values/test/channel/1", 0),
            ]
        )
        mock_client.unsubscribe.assert_called_once_with(
            [
                "vista/values/test/channel/1",
            ]
        )

        assert result == values[0] + 1

    @pytest.mark.parametrize(
        "test_input",
        [
            ([value for value in values]),
            ([np.array(value) for value in values]),
        ],
    )
    def test_MQTT_set_values_no_validation(self, mock_client, test_input):
        interface = Interface(
            host="testhost",
            port=1883,
            keepalive=60,
            poll_period=0.1,
            timeout=1,
            parallel=False,
        )

        interface.set_values(channels, test_input, configs={})

        # check all the calls
        assert len(mock_client.connect.call_args_list) == len(channels)
        assert len(mock_client.disconnect.call_args_list) == len(channels)

        # check the contents of the publish message
        for i in range(3):
            assert (
                json.loads(mock_client.publish.call_args_list[i][1]["payload"])["value"]
                == values[i]
            )
            assert (
                json.loads(mock_client.publish.call_args_list[i][1]["payload"])[
                    "channel"
                ]
                == f"test::channel:{i+1}"
            )
            assert (
                mock_client.publish.call_args_list[i][1]["topic"]
                == f"vista/set/test/channel/{i+1}/value"
            )

    def test_MQTT_set_values_with_validation(self, caplog, mock_client):
        interface = Interface(
            host="testhost",
            port=1883,
            keepalive=60,
            poll_period=0.1,
            timeout=1,
            parallel=False,
        )
        configs = {
            "test::channel:1": {
                "validate_readback": True,
                "readback_pv": "test::channel:1:read",
                "tolerance": 0.1,
                "count_down": 0.11,
                "offset": 0.2,
            }
        }

        with patch("mqtt.ValidationClient") as val_client:
            # with a value that is within the tolerance, we shouldn'get any
            # errors raised
            val_client.return_value.current_value = (
                values[0] - 0.2 + np.random.uniform(-0.05, 0.05)
            )
            interface.set_values(channels, values, configs=configs)

            # first check all the calls to the validation client - as we
            # have only configured one PV to be validated, these should
            # all only be called once
            val_client.return_value.connect.assert_called_once_with(
                "testhost", 1883, 60
            )
            val_client.return_value.disconnect.assert_called_once_with()
            val_client.return_value.loop_start.assert_called_once_with()
            val_client.return_value.loop_stop.assert_called_once_with()

        # if everything set correctly and within tolerance (as is in current_value)
        # then we should get no logged errors
        assert len(caplog.records) == 0

        # then check the calls to the publish client - here we should have as
        # many calls as there are channels
        assert len(mock_client.connect.call_args_list) == len(channels)
        assert len(mock_client.loop_stop.call_args_list) == len(channels)
        assert len(mock_client.disconnect.call_args_list) == len(channels)

    def test_MQTT_set_values_with_validation_timeout(self, caplog, mock_client):
        interface = Interface(
            host="testhost",
            port=1883,
            keepalive=60,
            poll_period=0.1,
            timeout=1,
            parallel=False,
        )
        configs = {
            "test::channel:1": {
                "validate_readback": True,
                "readback_pv": "test::channel:1:read",
                "tolerance": 0.1,
                "count_down": 0.11,
                "offset": 0.2,
            }
        }

        with patch("mqtt.ValidationClient") as val_client:
            # with a value that is within the tolerance, we shouldn'get any
            # errors raised
            val_client.return_value.current_value = 7
            interface.set_values(channels, values, configs=configs)

            # first check all the calls to the validation client - as we
            # have only configured one PV to be validated, these should
            # all only be called once
            val_client.return_value.connect.assert_called_once_with(
                "testhost", 1883, 60
            )
            val_client.return_value.disconnect.assert_called_once_with()
            val_client.return_value.loop_start.assert_called_once_with()
            val_client.return_value.loop_stop.assert_called_once_with()

        # although all the calls are correct, we should get an error logged that
        # the value wasn't set correctly
        assert len(caplog.records) == 1
        assert (
            caplog.records[0].getMessage()
            == "PV test::channel:1 (current: 7) cannot reach expected value (5.0) in designated time 0.11!"
        )
        # then check the calls to the publish client - here we should have as
        # many calls as there are channels
        assert len(mock_client.connect.call_args_list) == len(channels)
        assert len(mock_client.loop_stop.call_args_list) == len(channels)
        assert len(mock_client.disconnect.call_args_list) == len(channels)
