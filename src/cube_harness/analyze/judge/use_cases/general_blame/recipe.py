"""`general_blame` — the default judge recipe.

Output schema is identical to the legacy `JudgeOutput` (now `BaseJudgeOutput`),
preserving on-disk compatibility for the default judge path. Subclassing without
adding fields gives us a class identity (so the recipe can declare `output_model`)
without changing the JSON shape one bit.
"""

from __future__ import annotations

from cube_harness.analyze.judge.prompt import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT_TEMPLATE
from cube_harness.analyze.judge.recipe import BaseJudgeOutput, JudgeRecipe


class GeneralBlameOutput(BaseJudgeOutput):
    """The default judge output. Identical fields to `BaseJudgeOutput` —
    a class identity so the recipe can carry an `output_model` reference."""


RECIPE = JudgeRecipe(
    name="general_blame",
    system_prompt=JUDGE_SYSTEM_PROMPT,
    user_prompt_template=JUDGE_USER_PROMPT_TEMPLATE,
    output_model=GeneralBlameOutput,
    model="claude-sonnet-4-6",
)
