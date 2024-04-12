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

    @property
    def real(self):
        return self.raw.value


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
        interface = Interface(timeout=1, parallel=False, read_only=False)
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
        interface = Interface(timeout=1, parallel=False, read_only=False)
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
        interface = Interface(timeout=1, parallel=False, read_only=False)
        interface.set_values(dict(zip(channels, test_input)), configs={})
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

        configs = {
            # in reality this would be a partial function instead of a MagicMock
            "test::channel:1": MagicMock()
        }
        interface = Interface(timeout=1, parallel=False, read_only=False)
        interface.set_values(dict(zip(channels, values)), configs=configs)
        # check that the correct put values were called
        assert mock_context.return_value.put.call_args_list[0][0] == (
            channels[0],
            values[0],
        )
        assert mock_context.return_value.put.call_args_list[1][0] == (
            channels[1],
            values[1],
        )
        # check the validation function was called as expected
        configs["test::channel:1"].assert_called_once_with(
            set_pv="test::channel:1",
            set_value=5.0,
            context=mock_context.return_value,
            timeout=1,
        )
        assert len(configs["test::channel:1"].call_args_list) == 1
        # check that the context is closed correctly()
        assert len(mock_context.return_value.close.call_args_list) == len(channels)

    @pytest.mark.parametrize(
        "error",
        [(ValueError("validation function failed")), (TimeoutError("timeout error"))],
    )
    def test_epics_set_values_with_validation_fails(self, caplog, mock_context, error):
        mock_context.return_value.put = MagicMock()

        configs = {
            # in reality this would be a partial function instead of a MagicMock
            "test::channel:1": MagicMock(side_effect=error)
        }
        interface = Interface(timeout=1, parallel=False, read_only=False)
        interface.set_values(dict(zip(channels, values)), configs=configs)

        assert len(caplog.records) == 1
        assert caplog.records[0].getMessage() == str(error)

    @patch("time.sleep")
    def test_epics_set_values_put_timeout_retry_fails(
        self, mock_time, caplog, mock_context
    ):
        # here we test whether the put is called twice
        mock_context.return_value.put = MagicMock(
            side_effect=TimeoutError("timeout error")
        )

        interface = Interface(timeout=1, parallel=False, read_only=False)

        interface.set_values({channels[0]: values[0]}, configs={})

        # check that the put was attempted twice
        assert len(mock_context.return_value.put.call_args_list) == 2
        # make sure we waited in between
        mock_time.assert_called_once_with(1)

        # finally we log a message if both attempts were unsuccessful
        assert len(caplog.records) == 1
        assert (
            caplog.records[0].getMessage()
            == "Timeout on test::channel:1: timeout error after 2 attempts and a 1 second delay"
        )

    @patch("time.sleep")
    def test_epics_set_values_put_timeout_retry_successful(
        self, mock_time, caplog, mock_context
    ):
        mock_context.return_value.put = MagicMock(
            side_effect=[TimeoutError("timeout error"), None]
        )

        interface = Interface(timeout=1, parallel=False, read_only=False)

        with caplog.at_level(logging.DEBUG):
            interface.set_values({channels[0]: values[0]}, configs={})

        # check that the put was attempted twice
        assert len(mock_context.return_value.put.call_args_list) == 2
        # make sure we waited in between
        mock_time.assert_called_once_with(1)

        # finally we log a message if both attempts were unsuccessful and we should
        # also get a message about how long it takes to set the values
        assert len(caplog.records) == 2
        assert caplog.records[0].getMessage() == "put value 5.0 to test::channel:1"

    def test_epics_set_values_read_only(self, caplog, mock_context):
        caplog.set_level(logging.INFO)
        mock_context.return_value.put = MagicMock()
        interface = Interface(timeout=1, parallel=False, read_only=True)
        set_vals = dict(zip(channels, values))
        with caplog.at_level(logging.INFO):
            interface.set_values(set_vals, configs={})

        # check the log message is sent correctly and that context.put is not called
        mock_context.return_value.put.assert_not_called()
        assert len(caplog.records) == 1
        assert (
            caplog.records[0].getMessage()
            == f"Interface is set to read-only mode, cannot set {set_vals}"
        )

    def test_epics_set_value_read_only(self, caplog, mock_context):
        caplog.set_level(logging.INFO)
        mock_context.return_value.put = MagicMock()
        interface = Interface(timeout=1, parallel=False, read_only=True)
        with caplog.at_level(logging.INFO):
            interface.set_value(channels[0], values[0])

        # check the log message is sent correctly and that context.put is not called
        mock_context.return_value.put.assert_not_called()
        assert len(caplog.records) == 1
        assert (
            caplog.records[0].getMessage()
            == f"Interface is set to read-only mode, cannot set value {values[0]} to {channels[0]}"
        )
