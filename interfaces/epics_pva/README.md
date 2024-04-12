# EPICS PVAccess Interface for Badger
Interface for setting and reading EPICS PVAccess PV values utilising the [p4p](https://github.com/mdavidsaver/p4p) library. 

## Prerequisites

## Usage
Some general notes on usage:

### Read Only Mode
In some cases we may want to be testing our software stack on a live system but not want to be changing values. For these occasions we utilise a `read_only` flag configured in the constructor. Instead of actually setting values in the control system, the `set_values()` call will log a message to indicate the values that would have been set. `get_values()` will still return valid observations.

```python
interface = Interface(read_only=True)
```

### Retry on Timeout
In the case of a TimeoutError on a PV when doing a `put` call, we retry the call twice, waiting one second between. If this second attempt to set the PV fails, an exception is thrown. 

### Validation Functions
In the case of some PVs we may want to validate that the PV has in fact been set, for example by checking an associated readback PV. For these cases, we pass a `validation_function` to the `set_value()` call. It is expected that this function would be configured as a `functools.partial` function and multiple validation functions can be passed to the `set_values()` call as a dictionary with the key as the set PV name. An example is provided below:

Validation function:
```python
def wait_for_readback(
    set_pv: str, set_value: float, readback_pv: str, tolerance: float, count_down: int, offset: float, context: p4p.client.thread.Context, timeout: int
):
    time_limit = deepcopy(count_down)
    while count_down > 0:
        _value = context.get(readback_pv, timeout=timeout)
        if np.isclose(_value.real, set_value + offset,atol=tolerance):
            return _value.real
        else:
            time.sleep(0.1)
            count_down -= 0.1
    raise ValueError(
        f"PV {set_pv} (current {readback_pv}: {_value.real}) cannot reach expected value ({set_value + offset}) in designated time {time_limit}!"
    )
```

Expected usage:
```python
class Environment(badger.Environment):
    ...
    _configs: ClassVar[Dict[str, partial]] = {
        "EXAMPLE:PV:SET": partial(
            wait_for_readback,
            readback_pv="EXAMPLE:PV:READ",
            tolerance=0.05,
            count_down=15,
            offset=0,
        ),
    }
    def set_variables(self, variable_inputs):
        self.interface.set_values(
            variable_inputs,
            configs=self._configs,
        )
```


### Parallel Execution
When using these validation function, setting values on multiple PVs may cause long delays, espcially if the update rate of the READ PV is slow. Therefore we provide the option to run all of the `set` commands in parallel. This can be configured using the `parallel` flag in the constructor:
```python
interface = Interface(parallel=False)
```


## Testing
Initial tests for the interface are available in the `tests.py` file and can be run locally using `pytest tests.py` or `pytest --cov=. tests.py` from within the `epics_pva` directory.