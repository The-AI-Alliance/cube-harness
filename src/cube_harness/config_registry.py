"""Copy-on-access registry for canonical configs.

A recipe picks a canonical config by name and tweaks a few attributes:

    agent = GENNY_CONFIGS["swe"]
    agent.max_actions = 200

Each lookup returns a fresh deep copy, so a recipe can never mutate the
shared canonical instance (and corrupt every other recipe in the process).
The trade-off: you must bind to a variable before mutating —
``GENNY_CONFIGS["swe"].max_actions = 200`` mutates a throwaway and no-ops.

A ``Mapping`` (not a ``dict`` subclass) on purpose: ``dict.get`` / ``.values``
/ ``.items`` / ``**unpack`` would bypass ``__getitem__`` and hand out the
shared instance. ``Mapping`` routes every read through ``__getitem__``, so
copy-on-access actually holds.
"""

from collections.abc import Iterator, Mapping

from pydantic import BaseModel


class ConfigRegistry[T: BaseModel](Mapping[str, T]):
    """Maps a name to a canonical config; every lookup returns a deep copy."""

    def __init__(self, configs: dict[str, T]) -> None:
        self._configs = configs

    def __getitem__(self, name: str) -> T:
        try:
            return self._configs[name].model_copy(deep=True)
        except KeyError:
            raise KeyError(f"Unknown config {name!r}. Available: {sorted(self._configs)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._configs)

    def __len__(self) -> int:
        return len(self._configs)
