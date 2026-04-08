# BrowserGym Tool Refactor Plan

> Replaces the aspirational parts of `BROWSER_TOOL_DESIGN.md` with a concrete,
> scoped plan against today's codebase (2026-04-08).
>
> **This PR is not for merging** -- just for discussion and comments.

## Goal

Make `BrowsergymTool` a **thin, transparent wrapper** over BrowserGym's
`HighLevelActionSet` (actions) and observation utilities (obs). A developer
reading the code or the agent's tool descriptions should see bgym's native
action names, parameters, and descriptions -- not a cube-specific protocol layer.

## Non-goals (deferred)

- TaskLogic abstraction (tool-agnostic tasks) -- separate effort.
- CUATool / PlaywrightCodeTool -- separate effort.
- Changes to cube-standard's `BrowserTool` base class.

---

## Current State

```
BrowsergymTool(ToolWithTelemetry, BrowserTool, BidBrowserActionSpace)
      |                                              |
      |  inherits 12 @tool_action methods            |
      |  (browser_click, browser_type, ...)          |
      |                                              |
      |  Each method manually builds an action       |
      |  string like f'click(bid="{bid}")'           |
      |  and calls _execute_bgym_step()              |
      |                                              |
      v                                              v
  _execute_bgym_step(action_str)          BidBrowserActionSpace (ABC)
      |                                  in action_spaces/browser_action_space.py
      |  action_set.to_python_code()     -- 12 abstract @tool_action methods
      |  execute_python_code(code, page)    with cube-specific names
      v
  BrowserGym utilities
```

**Problems:**

1. **Extra protocol layer.** `BidBrowserActionSpace` defines 12 methods with
   cube-specific names (`browser_click`, `browser_type`, ...) that manually map
   to bgym action strings. This hides bgym's native names and misses actions
   (scroll, dblclick, focus, clear, upload_file, tab ops, send_msg_to_user).

2. **Not click-through.** A developer sees `browser_click(bid)` -> has to read
   the method body to discover it calls `click(bid=...)` in bgym. With bgym
   names exposed directly, IDE "go to definition" lands in bgym source.

3. **Error swallowing (PR #270).**
   - `report_infeasible_instructions` callback only logged -- agent saw "Success".
   - `last_action_error` never populated in obs dict (dead code in converter).
   - `send_msg_to_user` callback only logged -- never surfaced.

4. **Rigid action subset.** The 12-method protocol can't be reconfigured per
   benchmark. bgym already supports named subsets ("workarena", "webarena",
   "miniwob_all", etc.) -- we should use them.

---

## Target State

```
BrowsergymTool(ToolWithTelemetry, BrowserTool)
      |
      |  action_set property returns ActionSchema list
      |  built from HighLevelActionSet.to_tool_description()
      |
      |  execute_action() serialises Action -> bgym action string
      |  -> to_python_code() -> execute_python_code(code, page)
      |
      |  Captures: exceptions, infeasible reports, user messages
      |  Populates: last_action_error in obs dict
      v
  BrowserGym utilities (unchanged)
```

No `BidBrowserActionSpace`. No `BrowserActionSpace`. No manual method-per-action.

---

## Changes

### 1. Delete action space protocols

**Files to delete:**
- `src/cube_harness/action_spaces/browser_action_space.py`
- `src/cube_harness/action_spaces/__init__.py` (or make empty)

**Files to update (remove imports/inheritance):**
- `src/cube_harness/tools/browsergym.py` -- remove `BidBrowserActionSpace` from bases
- `src/cube_harness/tools/playwright.py` -- remove `BrowserActionSpace` from bases

> The same protocols also exist in `cube-standard/cube-browser-tool/`. That's a
> separate PR on cube-standard -- out of scope here, but should follow.

### 2. Rewrite BrowsergymTool action surface

Replace the 12 hand-written `@tool_action` methods with dynamic action
generation from bgym's `HighLevelActionSet`.

#### a) `action_set` property -- build from bgym

`HighLevelActionSet.to_tool_description(api="openai")` returns a list of dicts:
```python
[{
    "name": "click",
    "description": "Click an element.\n\nExamples:\n  click('a51')\n  ...",
    "parameters": {
        "type": "object",
        "properties": {
            "bid": {"type": "string", "description": "..."},
            "button": {"type": "string", "enum": ["left", "middle", "right"], ...},
            ...
        },
        "required": ["bid"]
    }
}, ...]
```

Convert each to `ActionSchema(name=..., description=..., parameters=...)`.
This gives the agent bgym's native names, full parameter schemas, and
descriptions -- with zero hand-coding.

#### b) `execute_action()` -- serialise back to bgym string

Override `execute_action(action: Action)` to:

1. Reconstruct the bgym action string from `action.name` + `action.arguments`:
   `f'{action.name}({", ".join(k=repr(v) for k,v in args.items())})'`
2. Call `_execute_bgym_step(action_str)` as today.
3. Append page observation.
4. Return `Observation`.

#### c) Keep checkbox fallback (for now)

The `browser_click` -> `_get_checkbox_state` -> `_toggle_checkbox_js` fallback
solves a real Playwright bug. Keep it as an internal hook in the click path:

```python
def _post_action_hook(self, action_name: str, action_args: dict, result: str) -> str:
    if action_name == "click":
        return self._checkbox_fallback(action_args["bid"], result)
    return result
```

This preserves the fix without needing a protocol method.

### 3. Fix error propagation (builds on PR #270)

In `_execute_bgym_step`:

- **`report_infeasible_instructions`**: Capture messages, return
  `"Failed (infeasible): ..."` to agent. *(done in PR #270)*
- **`send_msg_to_user`**: Capture messages, include in observation as a
  `Content` item named `"user_message"`. The agent needs to see these -- bgym
  tasks use `send_msg_to_user` for task-completion signals.
- **`last_action_error`**: Populate in obs dict from `_last_info`.
  *(done in PR #270)*

### 4. Make action subset configurable

`BrowsergymConfig` already has fields but not `action_subsets`. Add:

```python
class BrowsergymConfig(ToolConfig):
    action_subsets: list[str] = ["chat", "infeas", "bid", "nav", "tab"]
    # ... existing fields ...
```

Pass to `HighLevelActionSet(subsets=self.config.action_subsets)` in `__init__`.
Benchmarks set it via config:
```python
BrowsergymConfig(action_subsets=["workarena"])  # WorkArena's curated set
BrowsergymConfig(action_subsets=["bid", "nav", "tab", "chat", "infeas"])  # full BID set
```

### 5. Update tests

- `tests/test_browsergym.py` line 619: test that checks `action_set` contains
  all `BidBrowserActionSpace` methods -> replace with test that verifies
  `action_set` names match `HighLevelActionSet` action names.
- Add test: execute an action with invalid BID -> verify agent gets error string.
- Add test: `send_msg_to_user` -> verify message appears in observation.

### 6. Update docs and references

- `AGENTS.md` / `CLAUDE.md`: remove references to `BrowserActionSpace` / `BidBrowserActionSpace`.
- `docs/BROWSER_TOOL_DESIGN.md`: mark protocols as deleted (already noted there).
- `recipes/tool_api.py`: update MCP tool registration (no longer references `BrowserActionSpace`).

---

## Implementation Order

1. **PR #270** (submitted): Fix error propagation bugs. *(prerequisite)*
2. **PR A**: Delete `action_spaces/` directory, remove protocol inheritance
   from `BrowsergymTool` and `AsyncPlaywrightTool`. Add `action_subsets` config.
   Rewrite `BrowsergymTool` to use dynamic `action_set` from
   `HighLevelActionSet.to_tool_description()` and `execute_action()` override.
   Keep checkbox fallback as internal hook. Surface `send_msg_to_user`.
   Update tests.
3. **PR B** (cube-standard): Delete `BrowserActionSpace` / `BidBrowserActionSpace`
   from `cube-browser-tool`. Update `SyncPlaywrightTool` / `AsyncPlaywrightTool`.
4. **PR C**: Update docs, recipes, CLAUDE.md references.

---

## Open Questions

1. **`send_msg_to_user` as a tool action?** bgym includes it in the "chat"
   subset. Should the agent be able to call `send_msg_to_user("answer")`?
   If yes, it needs special handling -- not a browser action but a communication
   channel. Currently bgym tasks use it for the agent to report task completion.

2. **Checkbox fallback scope.** Is the Playwright click-doesn't-toggle bug
   still present in recent Playwright versions? If fixed upstream, we can
   delete the JS fallback entirely.

3. **`multiaction` support.** bgym's `HighLevelActionSet` supports
   `multiaction=True` (multiple actions per step). cube-standard already
   supports multiple actions per `AgentOutput`. Should we enable this?
   Default bgym behavior is `multiaction=True`.
