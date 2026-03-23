# OSWorld Recipes

## eval_osworld.py — quick / cheap eval

Genny agent with GPT-5 mini, axtree-only observations, 15 actions max. No screenshots.
Good for fast iteration and smoke-testing changes without burning tokens.

```bash
uv run recipes/osworld/eval_osworld.py debug   # 2 tasks, sequential
uv run recipes/osworld/eval_osworld.py         # full eval, 3 workers
```

## haiku.py — leaderboard config

Genny agent with Claude Haiku, multimodal observations (screenshot + axtree), 3-step
rolling context, 100 actions max. Produces strong leaderboard results.

```bash
uv run recipes/osworld/haiku.py debug   # debug_tasks.json, sequential
uv run recipes/osworld/haiku.py         # test_small (no gdrive), 3 workers
```
