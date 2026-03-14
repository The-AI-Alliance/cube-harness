# Configuration in cube-harness

## The core idea: recipes are your config files

If you come from tools like Hydra, Click, or JSON/YAML config files, the pattern here will feel different — but deliberately so.

In cube-harness, **a recipe is your configuration**. There is no separate config file to manage, no CLI flags to remember, and no schema to learn. You copy a recipe, edit it in Python, and run it. The recipe *is* the experiment specification.

```python
# recipes/my_experiment.py  <-- this is your config file

llm_config    = LLMConfig(model_name="gpt-4.5-mini", temperature=0.0)
agent_config  = ReactAgentConfig(llm_config=llm_config)
tool_config   = PlaywrightConfig(use_screenshot=True, headless=True)
benchmark     = MiniWobBenchmark(default_tool_config=tool_config)

exp = Experiment(
    name="my_experiment",
    output_dir=output_dir,
    agent_config=agent_config,
    benchmark=benchmark,
    max_steps=10,
)
```

To run a different experiment, copy the file and change what you need. Version control tracks the diff. No hidden state.

## Config objects are typed and serializable

Every `*Config` class is a [Pydantic](https://docs.pydantic.dev/) model. This gives you three things for free:

**1. Type safety at definition time.** Invalid configs fail immediately with a clear error, not silently at step 500.

**2. Full serialization and deserialization.** Every experiment serializes its full config to disk alongside results. You can always recover exactly what ran:

```python
config = ReactAgentConfig.model_validate_json(path.read_text())
```

Types are preserved on round-trip — no manual casting, no string-to-enum conversions.

**3. Composability.** Configs nest naturally. `ReactAgentConfig` holds an `LLMConfig`. `Experiment` holds an `AgentConfig` and a `Benchmark`. The hierarchy is explicit in the code.

## Why not Hydra / YAML / CLI flags?

| Approach | Problem |
|---|---|
| YAML / JSON files | No types. Errors surface at runtime. Hard to compose or refactor. |
| Hydra | Powerful but adds significant complexity: decorators, config groups, overrides syntax, interpolation. Overkill for most experiments. |
| CLI flags | Good for a handful of options. Breaks down with nested configs (agent inside experiment inside runner). |
| **Recipes (this approach)** | Full Python expressiveness. Refactor with your IDE. Type-checked. Diff-friendly. No new syntax to learn. |

The tradeoff: you can't run a sweep from the command line with `+llm.temperature=0.5`. For sweeps, write a loop in Python — it's two lines and you have full control.

## Running sweeps

```python
for temperature in [0.0, 0.5, 1.0]:
    exp = Experiment(
        name=f"sweep_temp_{temperature}",
        agent_config=ReactAgentConfig(
            llm_config=LLMConfig(model_name="gpt-4.5-mini", temperature=temperature)
        ),
        ...
    )
    run_with_ray(exp)
```

## Getting started

Copy an existing recipe from [`recipes/`](../recipes/) and modify it. The built-in configs are:

| Class | What it configures |
|---|---|
| `LLMConfig` | Model name, temperature, max tokens, provider |
| `ReactAgentConfig` | Agent loop behavior, LLM config |
| `PlaywrightConfig` | Browser tool: headless, screenshots, HTML pruning |
| `Experiment` | Name, output dir, agent, benchmark, max steps |

All fields are documented in the class docstrings.
