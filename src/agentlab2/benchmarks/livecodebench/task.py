"""LiveCodeBench task implementation."""

import base64
import json
import logging
import pickle
import zlib
from typing import Any

from agentlab2.action_spaces.swe_action_space import SWEActionSpace
from agentlab2.core import ActionSchema, ActionSubset, Observation, Task
from agentlab2.tools.daytona import DaytonaSWETool

logger = logging.getLogger(__name__)


class LiveCodeBenchTask(Task):
    """A single LiveCodeBench coding task."""

    validate_per_step: bool = False
    supported_actions: ActionSubset = (
        SWEActionSpace.bash,
        SWEActionSpace.read_file,
        SWEActionSpace.write_file,
    )
    _tool: DaytonaSWETool

    def __init__(
        self,
        id: str,
        question_title: str,
        question_content: str,
        platform: str,
        difficulty: str,
        starter_code: str,
        public_test_cases: str,
        private_test_cases: str,
        metadata: str,
    ) -> None:
        self.id = id
        self.question_title = question_title
        self.question_content = question_content
        self.platform = platform
        self.difficulty = difficulty
        self.starter_code = starter_code
        self.public_test_cases = public_test_cases
        self.private_test_cases = private_test_cases
        self.metadata = metadata
        self._solution_file = "/workspace/solution.py"

    def setup(self, tool: DaytonaSWETool) -> tuple[Observation, dict]:
        """Initialize the task environment."""
        logger.info(f"Setting up LiveCodeBench task: {self.question_title}")

        self._tool = tool
        self._tool.bash("mkdir -p /workspace")
        if self.starter_code:
            self._tool.write_file(self._solution_file, self.starter_code)

        # Build task prompt
        prompt = self._build_prompt()
        obs = Observation.from_text(prompt)

        return obs, {
            "task_id": self.id,
            "title": self.question_title,
            "platform": self.platform,
            "difficulty": self.difficulty,
        }

    def _build_prompt(self) -> str:
        """Build the task prompt for the agent."""
        # Show public test cases as examples (validation uses private tests)
        public_tests = self._parse_test_cases(self.public_test_cases)
        test_examples = ""
        if public_tests:
            test_examples = "\n\nExample test cases:"
            for i, tc in enumerate(public_tests, 1):
                test_examples += f"\nInput {i}: {tc.get('input', '')}"
                test_examples += f"\nExpected Output {i}: {tc.get('output', '')}"

        starter = ""
        if self.starter_code:
            starter = f"\n\nStarter code (saved to {self._solution_file}):\n```python\n{self.starter_code}\n```"

        return f"""# {self.question_title}

{self.question_content}
{test_examples}
{starter}

Your task:
1. Write a Python solution to the problem
2. Save it to {self._solution_file}
3. Test it with the provided examples
4. Call final_step when done

The solution should read from stdin and print to stdout."""

    def _parse_test_cases(self, test_cases_str: str) -> list[dict[str, Any]]:
        """Parse test cases from JSON string or encoded format.

        LiveCodeBench uses two formats:
        - public_test_cases: plain JSON string
        - private_test_cases: base64 -> zlib -> pickle -> JSON
        """
        if not test_cases_str:
            return []
        # Try plain JSON first (public test cases)
        try:
            return json.loads(test_cases_str)
        except json.JSONDecodeError:
            pass
        # Try encoded format (private test cases): base64 -> zlib -> pickle -> json
        try:
            decoded = base64.b64decode(test_cases_str)
            decompressed = zlib.decompress(decoded)
            unpickled = pickle.loads(decompressed)
            return json.loads(unpickled)
        except Exception as e:
            logger.warning(f"Failed to decode test cases: {e}")
            return []

    def _extract_stdout(self, bash_output: str) -> str:
        """Extract stdout from structured bash output, ignoring stderr and metadata."""
        lines = []
        for line in bash_output.split("\n"):
            # Skip our tool's metadata markers
            if line.startswith("[stderr]") or line.startswith("[exit_code:") or line.startswith("[error]"):
                break  # Everything after these markers is not stdout
            if line.startswith("bash:"):
                continue  # Skip bash warnings
            lines.append(line)
        return "\n".join(lines).strip()

    def validate_task(self, obs: Observation, *args) -> tuple[float, dict]:
        """Validate the solution against test cases."""
        # Check if solution file exists
        solution = self._tool.read_file(self._solution_file)
        if "Error reading" in solution:
            return 0.0, {"done": True, "reason": "No solution file found", "passed": 0, "total": 0}

        # Run against private test cases (hidden from the agent)
        # Fall back to public only if private tests are unavailable
        test_cases = self._parse_test_cases(self.private_test_cases)
        if not test_cases:
            logger.warning("No private test cases found, falling back to public tests")
            test_cases = self._parse_test_cases(self.public_test_cases)

        if not test_cases:
            return 0.0, {"done": True, "reason": "No test cases available", "passed": 0, "total": 0}

        passed = 0
        total = len(test_cases)
        results = []

        for i, tc in enumerate(test_cases):
            test_input = tc.get("input", "")
            expected_output = tc.get("output", "").strip()

            # Write test input to file
            self._tool.write_file("/workspace/test_input.txt", test_input)

            # Run solution (stderr captured separately by tool)
            result = self._tool.bash(
                "cd /workspace && timeout 10 python solution.py < test_input.txt",
                timeout=15,
            )

            # Extract stdout from structured output
            actual_output = self._extract_stdout(result)

            # Check if output matches
            test_passed = actual_output == expected_output
            if test_passed:
                passed += 1

            results.append(
                {
                    "test_idx": i,
                    "passed": test_passed,
                    "expected": expected_output[:100],
                    "actual": actual_output[:100],
                }
            )

        reward = passed / total if total > 0 else 0.0
        return reward, {
            "done": True,
            "passed": passed,
            "total": total,
            "pass_rate": reward,
            "results": results[:5],  # First 5 results for debugging
        }

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Filter to only SWE actions."""
        supported_names = {action.__name__ for action in self.supported_actions}
        filtered = [a for a in actions if a.name in supported_names]
        logger.info(f"Filtered to {len(filtered)} SWE actions for LiveCodeBench")
        return filtered

    def finished(self) -> bool:
        """Check if task is complete (after validation)."""
        return False  # Let the environment handle this via final_step
