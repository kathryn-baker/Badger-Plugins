import sys
from pathlib import Path

current_file = Path(__file__)
root_dir = current_file.parents[1]
sys.path.insert(0, str(root_dir))
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from p4p.client.thread import TimeoutError

channels = ["test::channel:1", "test::channel:2"]
values = [5.0, 6.0]

from epics_pva import Interface


class MockRaw:
    def __init__(self, value) -> None:
        self.value = value


class p4pValue:
    """Mock class designed to replicate the p4p Value class returned from `get` calls"""

    def __init__(self, value) -> None:
        self.raw = MockRaw(value)


@pytest.fixture
def mock_context():
    with patch("epics_pva.Context") as ctx:
        yield ctx


class TestEPICSInterface:
    def test_epics_get_values(self, mock_context):
        mock_context.return_value.get.return_value = [
            p4pValue(values[0]),
            p4pValue(values[1]),
        ]
        interface = Interface(poll_period=0.1, timeout=1, parallel=False)
        result = interface.get_values(channels)
        assert result == {
            "test::channel:1": values[0],
            "test::channel:2": values[1],
        }
        mock_context.return_value.close.assert_called_once_with()

    def test_epics_get_values_timeout(self, caplog, mock_context):
        # we get a timeout error first, then we get the individual values
        mock_context.return_value.get.side_effect = [
            TimeoutError("original error"),
            TimeoutError("individual timeout"),
            p4pValue(values[1]),
        ]
        interface = Interface(poll_period=0.1, timeout=1, parallel=False)
        result = interface.get_values(channels)

        # first check the returned values
        assert np.isnan(result["test::channel:1"])
        assert result["test::channel:2"] == values[1]

        # then check the logs
        assert len(caplog.records) == 2
        assert (
            caplog.records[0].getMessage()
            == f"Timeout error from {channels}, retrying individually"
        )
        assert caplog.records[1].getMessage() == f"{channels[0]}: individual timeout"

        assert len(mock_context.return_value.close.call_args_list) == len(channels) + 1

    @pytest.mark.parametrize(
        "test_input",
        [
            ([value for value in values]),
            ([np.array(value) for value in values]),
        ],
    )
    def test_epics_set_values_no_validation(self, mock_context, test_input):
        mock_context.return_value.put = MagicMock()
        interface = Interface(poll_period=0.1, timeout=1, parallel=False)
        interface.set_values(channels, test_input, configs={})
        assert mock_context.return_value.put.call_args_list[0][0] == (
            channels[0],
            values[0],
        )
        assert mock_context.return_value.put.call_args_list[1][0] == (
            channels[1],
            values[1],
        )
        assert len(mock_context.return_value.close.call_args_list) == len(channels)

    def test_epics_set_values_with_validation(self, mock_context):
        mock_context.return_value.put = MagicMock()
        # we only configure one of the values to have validation so we only need to
        # set values of the mock here
        mock_context.return_value.get = MagicMock(
            side_effect=[
                p4pValue(7),
                p4pValue(values[0] - 0.2 + np.random.uniform(-0.05, 0.05)),
            ]
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
        interface = Interface(poll_period=0.1, timeout=1, parallel=False)
        interface.set_values(channels, values, configs=configs)
        # check that the correct put values were called
        assert mock_context.return_value.put.call_args_list[0][0] == (
            channels[0],
            values[0],
        )
        assert mock_context.return_value.put.call_args_list[1][0] == (
            channels[1],
            values[1],
        )
        # then check that the correct readback PV was used were called
        for mock_call in mock_context.return_value.get.call_args_list:
            assert mock_call[0][0] == "test::channel:1:read"
        assert len(mock_context.return_value.close.call_args_list) == len(channels)

    def test_epics_set_values_validation_timeout(self, caplog, mock_context):
        mock_context.return_value.put = MagicMock()
        # we only configure one of the values to have validation so we only need to
        # set values of the mock here
        mock_context.return_value.get = MagicMock(
            # in this case, neither value satisfies the conditions we've placed
            side_effect=[p4pValue(7), p4pValue(7.01)]
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
        interface = Interface(poll_period=0.1, timeout=1, parallel=False)
        interface.set_values(channels, values, configs=configs)

        # check that the correct put values were called
        assert mock_context.return_value.put.call_args_list[0][0] == (
            channels[0],
            values[0],
        )
        assert mock_context.return_value.put.call_args_list[1][0] == (
            channels[1],
            values[1],
        )
        # check the error is logged correctly
        assert len(caplog.records) == 1
        assert (
            caplog.records[0].getMessage()
            == f"PV {channels[0]} (current: 7.01) cannot reach expected value ({float(values[0])}) in designated time 0.11!"
        )
        assert len(mock_context.return_value.close.call_args_list) == len(channels)

    def test_epics_set_values_read_only(self, caplog, mock_context):
        caplog.set_level(logging.INFO)
        mock_context.return_value.put = MagicMock()
        interface = Interface(
            poll_period=0.1, timeout=1, parallel=False, read_only=True
        )
        interface.set_values(channels, values, configs={})

        # check the log message is sent correctly and that context.put is not called
        mock_context.return_value.put.assert_not_called()
        assert len(caplog.records) == 1
        assert (
            caplog.records[0].getMessage()
            == f"Interface is set to read-only mode, cannot set values {values} to {channels}"
        )

    def test_epics_set_value_read_only(self, caplog, mock_context):
        caplog.set_level(logging.INFO)
        mock_context.return_value.put = MagicMock()
        interface = Interface(
            poll_period=0.1, timeout=1, parallel=False, read_only=True
        )
        interface.set_value(channels[0], values[0], set_config={})

        # check the log message is sent correctly and that context.put is not called
        mock_context.return_value.put.assert_not_called()
        assert len(caplog.records) == 1
        assert (
            caplog.records[0].getMessage()
            == f"Interface is set to read-only mode, cannot set value {values[0]} to {channels[0]}"
        )
