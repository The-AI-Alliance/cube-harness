"""Tests for Python action set."""

import pytest

from agentlab2.bgym_core.action.python import PythonActionSet

ACTIONS_TO_TEST = [
    (
        """\
a = 0
""",
        """\
a = 0
""",
    ),
    (
        """\
```
a = 0
```
""",
        """\
a = 0
""",
    ),
    (
        """\
```python
a = 0
```
""",
        """\
a = 0
""",
    ),
    (
        """\
```python
a = 0
```
This is an explanation
```python
b = 3
```
More explanations
""",
        """\
a = 0

b = 3
""",
    ),
]


@pytest.mark.parametrize("action,expected_code", ACTIONS_TO_TEST)
def test_action_cleaning(action: str, expected_code: str) -> None:
    action_set = PythonActionSet()
    code = action_set.to_python_code(action)

    assert code == expected_code
