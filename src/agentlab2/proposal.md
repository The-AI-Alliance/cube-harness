# SWE Sandbox Proposal

## Option A: Direct Tool (like PlaywrightTool)

```
Agent → SWEActionSpace → DaytonaSWETool (Daytona logic embedded)
```

```python
class DaytonaSWETool(Tool, SWEActionSpace):
    action_space = SWEActionSpace
    
    def __init__(self, config: DaytonaSWEToolConfig):
        self._client = DaytonaClient()  # Daytona embedded
        self._sandbox = None
    
    def bash(self, cmd: str) -> str:
        return self._sandbox.exec(cmd)  # Direct Daytona call
```

**Pros:** Simpler, fewer abstractions, faster to build  
**Cons:** Coupled to Daytona, need separate `DockerSWETool`, `LocalSWETool` classes

---

## Option B: Sandbox Protocol (Recommended)

```
Agent → SWEActionSpace → SWETool → Sandbox(Protocol) → DaytonaSandbox/DockerSandbox/Local
```

```python
class Sandbox(Protocol):
    def start(self) -> None: ...
    def exec(self, cmd: str, cwd: str | None = None, timeout: int | None = None) -> ExecResult: ...
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...
    def close(self) -> None: ...
```

```python
class SWETool(Tool, SWEActionSpace):
    action_space = SWEActionSpace
    
    def __init__(self, sandbox: Sandbox):  # Injected
        self.sandbox = sandbox
    
    def bash(self, cmd: str) -> str:
        return self.sandbox.exec(cmd)  # Delegates to any backend
```

**Pros:** Swappable backends, testable (mock sandbox), single `SWETool` class  
**Cons:** Extra abstraction layer, more upfront work

---

## Tradeoffs

| Aspect | Option A (Direct) | Option B (Protocol) |
|--------|-------------------|---------------------|
| Complexity | Lower | Medium |
| Backend swapping | New tool class per backend | Config change |
| Local debugging | Needs `LocalSWETool` | Use `LocalSandbox` |
| Testing | Mock entire tool | Mock `Sandbox` only |
| Code duplication | High (per backend) | Low (one `SWETool`) |

---

## Usage (Option B)

```python
benchmark = SWEBenchmark(
    tool_config=SWEToolConfig(
        sandbox_config=DaytonaConfig(image="python:3.13", cpus=4)
        # OR: DockerConfig(dockerfile="./Dockerfile")
        # OR: LocalConfig()  # for debugging
    )
)
```

---

## Note on PlaywrightTool

Same pattern could apply: `BrowserActionSpace` + `Browser(Protocol)` with `PlaywrightBrowser`, `BrowserBaseBrowser`, etc. Currently coupled, could be refactored.
