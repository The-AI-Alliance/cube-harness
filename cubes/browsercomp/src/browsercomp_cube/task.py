"""BrowseCompTask and BrowseCompTaskConfig for BrowseComp benchmark."""

import re
from typing import Any

import litellm

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Toolbox, ToolboxConfig

from browsercomp_cube.tool import SubmitAnswerTool, SubmitAnswerToolConfig
from cube_web_tool import BraveWebSearchToolConfig, WebFetchToolConfig

_GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous \
[correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' \
if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only \
on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on \
any background to the problem, do not attempt to solve the problem, do not argue for any answer different than \
[correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small \
margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, \
ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no \
confidence score available."""

_FORMAT_INSTRUCTIONS = (
    "\n\nPlease structure your final answer as:\n"
    "Explanation: <your reasoning>\n"
    "Exact Answer: <the precise answer>\n"
    "Confidence: <0-100>"
)


class BrowseCompTask(Task):
    """A single BrowseComp information-retrieval task."""

    validate_per_step: bool = False
    accept_agent_stop: bool = True
    grader_retries: int = 3
    scorer_model: str = "gpt-5.4-mini"

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        problem = self.metadata.extra_info["problem"]
        prompt = problem + _FORMAT_INSTRUCTIONS
        return Observation.from_text(prompt), {"problem": problem}

    def _call_grader(self, prompt: str, scorer_model: str) -> bool:
        completion = litellm.completion(
            model=scorer_model,
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content or ""
        match = re.search(r"correct:\s*(yes|no)", response, re.IGNORECASE)
        if not match:
            raise ValueError(f"Grader response missing 'correct: yes/no':\n{response}")
        return match.group(1).lower() == "yes"

    def _submit_tool(self) -> SubmitAnswerTool:
        assert isinstance(self.tool, Toolbox)
        tool = self.tool.find_tool(SubmitAnswerTool)
        assert isinstance(tool, SubmitAnswerTool)
        return tool

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        submitted = self._submit_tool().last_answer
        if submitted is None:
            return 0.0, {"correct": False, "submitted": None, "reason": "No answer submitted"}

        question = self.metadata.extra_info["problem"]
        correct_answer = self.metadata.extra_info["answer"]
        prompt = _GRADER_TEMPLATE.format(
            question=question,
            response=submitted,
            correct_answer=correct_answer,
        )

        last_error: Exception | None = None
        for _ in range(self.grader_retries):
            try:
                is_correct = self._call_grader(prompt, self.scorer_model)
                return (1.0 if is_correct else 0.0), {"correct": is_correct, "submitted": submitted}
            except Exception as e:
                last_error = e

        return 0.0, {"correct": False, "submitted": submitted, "grader_error": str(last_error)}

    def finished(self, obs: Observation | None = None) -> bool:
        return self._submit_tool().last_answer is not None


class BrowseCompTaskConfig(TaskConfig):
    """Serializable configuration that produces a BrowseCompTask."""

    scorer_model: str = "gpt-5.4-mini"

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> BrowseCompTask:
        from browsercomp_cube.benchmark import BrowseCompBenchmark

        task_metadata: TaskMetadata = BrowseCompBenchmark.task_metadata[self.task_id]
        tool_cfg = self.tool_config or ToolboxConfig(
            tool_configs=[BraveWebSearchToolConfig(), WebFetchToolConfig(), SubmitAnswerToolConfig()]
        )
        return BrowseCompTask(
            metadata=task_metadata,
            tool_config=tool_cfg,
            scorer_model=self.scorer_model,
            container_backend=container_backend,
        )
