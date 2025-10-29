"""
Test Generation Module for RL Self-Repair

Generates synthetic tests from problem specs to drive the RL repair loop without
using ground-truth tests (avoids test leakage).
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Iterable
from dataclasses import dataclass

from lcb_runner.lm_styles import LanguageModel, LMStyle  # assumes LMStyle exists


CATEGORIES = {"basic", "boundary", "edge", "adversarial"}


@dataclass
class TestCase:
    """Represents a single test case with input/output and metadata."""
    input_val: Any
    expected_output: Any
    confidence: float               # 0..1
    kind: str                       # "exact", "property", "approximate"
    category: str                   # one of CATEGORIES
    rationale: Optional[str] = None


class SyntheticTestGenerator:
    """
    Generates synthetic test cases from problem specifications.
    Uses LLM to create a test plan and concrete test I/O pairs with self-consistency checks.
    """
    def __init__(self, model: LanguageModel, base_runner, deterministic_decoding: bool = True):
        self.model = model
        self.base_runner = base_runner
        self.deterministic_decoding = deterministic_decoding  # e.g., temperature=0 inside base_runner if supported

    def generate_tests(
        self,
        question: str,
        code_type: str = "stdin",   # "stdin" or "call_based"
        num_tests: int = 5,
        self_consistency_checks: int = 3,
        min_confidence: float = 0.5,
        progress_bar=None
    ) -> List[TestCase]:
        """
        Generate synthetic test cases for a problem.
        """
        if progress_bar:
            progress_bar.set_description("Generating test plan...")
        plan_raw = self._generate_test_plan(question, code_type)
        
        if progress_bar:
            progress_bar.set_description("Generating concrete tests...")
        raw_tests = self._generate_concrete_tests(question, plan_raw, num_tests, code_type)
        
        if progress_bar:
            progress_bar.set_description(f"Validated tests...")
        validated_tests = self._validate_with_self_consistency(
            question, raw_tests, self_consistency_checks, code_type, min_confidence
        )
        
        if progress_bar:
            progress_bar.set_description(f"Generated {len(validated_tests)}/{len(raw_tests)} validated tests")
            progress_bar.update(1)
        
        return validated_tests

    # ---------- LLM calls ----------

    def _is_chat_style(self) -> bool:
        ms = getattr(self.model, "model_style", None)
        if ms is None:
            return False
        # Prefer explicit enum checks if available
        chat_styles = {
            getattr(LMStyle, "OpenAIChat", None),
            getattr(LMStyle, "DeepSeekAPI", None),
            getattr(LMStyle, "Claude", None),
            getattr(LMStyle, "Claude3", None),
            getattr(LMStyle, "LLaMa3", None),
        }
        return ms in chat_styles

    def _llm_respond(self, user_content: str, system_content: Optional[str] = None) -> str:
        """
        Single entrypoint to call the base_runner. If your base_runner supports deterministic decoding,
        wire deterministic_decoding here (e.g., temperature=0).
        """
        if self._is_chat_style():
            messages = []
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": user_content})
            # If your base_runner supports per-call params, pass temperature=0 when deterministic
            outputs = self.base_runner.prompts_to_outputs([messages])
        else:
            outputs = self.base_runner.prompts_to_outputs([user_content])
        return outputs[0][0] if outputs and outputs[0] else ""

    def _generate_test_plan(self, question: str, code_type: str) -> str:
        prompt = f"""Given this problem specification, list a compact test plan:
1) Basic/nominal cases
2) Boundary conditions
3) Edge cases
4) Adversarial/random cases

For each category, describe the intent and 2-3 representative inputs.

Problem:
{question}

Code type: {code_type}

Provide the test plan in JSON format with fields: category, intent, example_inputs.
"""
        response = self._llm_respond(
            user_content=prompt,
            system_content="You are an expert at designing comprehensive test cases for programming problems."
        )
        logging.info(f"Generated test plan (truncated): {response[:200]}...")
        return response

    def _generate_concrete_tests(
        self,
        question: str,
        test_plan: str,
        num_tests: int,
        code_type: str
    ) -> List[Dict[str, Any]]:
        prompt = f"""Based on this test plan, instantiate concrete (input, expected output) pairs.
Compute the expected outputs exactly. If non-trivial, show the minimal reasoning steps.

Problem:
{question}

Test Plan:
{test_plan}

Generate {num_tests} diverse test cases covering all categories.

Provide as a JSON array with objects containing:
- category: basic/boundary/edge/adversarial
- input: the input value (as string or JSON)
- expected: the expected output (as string or JSON)
- rationale: brief explanation

Code type: {code_type}

IMPORTANT: Return ONLY a valid JSON array, no markdown, no code blocks, no extra text.
Example format:
[
  {{"category": "basic", "input": "test", "expected": "result", "rationale": "..."}},
  {{"category": "boundary", "input": "edge", "expected": "output", "rationale": "..."}}
]
"""
        response = self._llm_respond(
            user_content=prompt,
            system_content="You are an expert at computing exact expected outputs for test cases. Return ONLY JSON, no other text."
        )
        
        tests = self._parse_json_array(response)
        if not tests:
            logging.info(f"Failed to parse JSON, attempting manual extraction...")
            # Try to manually extract test-like structures
            tests = self._manual_extract_tests(response)
            
        if not tests:
            logging.warning("All parsing attempts failed. Falling back to a minimal test.")
            return [{
                "category": "basic",
                "input": "",
                "expected": "",
                "rationale": "Fallback test due to parsing error"
            }]

        # Validate and coerce
        valid = []
        for t in tests:
            cat = str(t.get("category", "basic")).lower()
            if cat not in CATEGORIES:
                cat = "basic"
            if "input" not in t or "expected" not in t:
                continue
            valid.append({
                "category": cat,
                "input": t["input"],
                "expected": t["expected"],
                "rationale": t.get("rationale", "")
            })
        print(f"Generated {len(valid)} concrete tests")
        return valid

    # ---------- Self-consistency ----------

    def _validate_with_self_consistency(
        self,
        question: str,
        raw_tests: List[Dict[str, Any]],
        num_checks: int,
        code_type: str,
        min_confidence: float
    ) -> List[TestCase]:
        validated: List[TestCase] = []
        for idx, test in enumerate(raw_tests):
            baseline, confidence = self._compute_expected_with_consistency(
                question, test["input"], num_checks, code_type
            )
            if confidence >= min_confidence:
                validated.append(TestCase(
                    input_val=test["input"],
                    expected_output=baseline,
                    confidence=confidence,
                    kind="exact",
                    category=test.get("category", "basic"),
                    rationale=test.get("rationale", "")
                ))
        print(f"Validated {len(validated)}/{len(raw_tests)} tests")
        return validated

    def _compute_expected_with_consistency(
        self,
        question: str,
        input_val: Any,
        num_checks: int,
        code_type: str
    ) -> Tuple[Any, float]:
        outputs: List[str] = []
        for _ in range(max(1, num_checks)):
            prompt = f"""Given this problem:
{question}

And this test input:
{json.dumps(input_val)}

Compute the expected output exactly. Show your reasoning briefly, then give the final answer.
Code type: {code_type}

Only return the expected output as a single value (no extra text).
"""
            out = self._llm_respond(
                user_content=prompt,
                system_content="You are an expert at computing exact expected outputs."
            )
            outputs.append(self._strip_fences(out).strip())

        baseline = outputs[0]
        agree = sum(1 for o in outputs[1:] if self._outputs_match(baseline, o))
        confidence = (agree + 1) / max(1, num_checks)
        return baseline, confidence

    # ---------- Utilities ----------

    def _strip_fences(self, text: str) -> str:
        if "```" not in text:
            return text
        # Try ```json fenced block first
        m = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1)
        # Any fenced block
        m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1)
        return text

    def _parse_json_array(self, text: str) -> List[Dict[str, Any]]:
        """
        Try multiple ways to extract a top-level JSON array from text.
        """
        candidate = self._strip_fences(text).strip()
        
        # Direct parse
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "tests" in obj and isinstance(obj["tests"], list):
                return obj["tests"]
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

        # Heuristic: find first '[' ... matching ']' at same nesting level
        start = candidate.find('[')
        end = candidate.rfind(']')
        if 0 <= start < end:
            frag = candidate[start:end+1]
            try:
                obj = json.loads(frag)
                if isinstance(obj, list):
                    return obj
            except json.JSONDecodeError:
                pass
        
        # Try to fix common issues
        # Remove trailing commas
        fixed = re.sub(r',\s*}', '}', candidate)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        try:
            obj = json.loads(fixed)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

        return []
    
    def _manual_extract_tests(self, response: str) -> List[Dict[str, Any]]:
        """
        Manually extract test-like structures from response as a fallback.
        """
        # Look for patterns like "input:", "expected:", "category:"
        lines = response.split('\n')
        tests = []
        current_test = {}
        
        for line in lines:
            line = line.strip()
            if 'category' in line.lower() and ':' in line:
                if current_test and 'input' in current_test and 'expected' in current_test:
                    tests.append(current_test)
                current_test = {'category': 'basic', 'rationale': ''}
                parts = line.split(':', 1)
                if len(parts) > 1:
                    cat = parts[1].strip().replace('"', '').replace("'", "").lower()
                    if cat not in CATEGORIES:
                        cat = 'basic'
                    current_test['category'] = cat
            elif 'input' in line.lower() and ':' in line and 'expected' not in line:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    # Extract the input value, handling escaped newlines
                    input_val = parts[1].strip()
                    # Remove surrounding quotes
                    if input_val.startswith('"') and input_val.endswith('"'):
                        input_val = input_val[1:-1]
                    # Handle escaped newlines
                    input_val = input_val.replace('\\n', '\n')
                    current_test['input'] = input_val
            elif 'expected' in line.lower() and ':' in line:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    # Extract expected value
                    expected_val = parts[1].strip()
                    # Remove surrounding quotes
                    if expected_val.startswith('"') and expected_val.endswith('"'):
                        expected_val = expected_val[1:-1]
                    current_test['expected'] = expected_val
            elif 'rationale' in line.lower() and ':' in line:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    current_test['rationale'] = parts[1].strip().replace('"', '').replace("'", "")
        
        if current_test and 'input' in current_test and 'expected' in current_test:
            tests.append(current_test)
        
        return tests if tests else []

    def _outputs_match(self, output1: str, output2: str, float_tol: float = 1e-6) -> bool:
        """
        Compare two outputs with normalization:
        - Try JSON equality
        - Try float-aware scalar equality
        - Fallback to normalized string equality
        """
        o1 = str(output1).strip()
        o2 = str(output2).strip()

        # Try JSON equality
        try:
            j1 = json.loads(o1)
            j2 = json.loads(o2)
            return j1 == j2
        except (json.JSONDecodeError, ValueError):
            pass

        # Try float-aware numeric comparison
        try:
            f1 = float(o1)
            f2 = float(o2)
            return abs(f1 - f2) <= float_tol
        except ValueError:
            pass

        # Normalize whitespace and simple list formatting
        def norm(s: str) -> str:
            s = s.replace("\r", "").strip()
            s = re.sub(r"\s+", " ", s)
            s = s.replace(", ", ",")
            return s

        return norm(o1) == norm(o2)


def generate_property_tests(
    question: str,
    model: LanguageModel,
    base_runner,
    num_properties: int = 2
) -> List[TestCase]:
    """
    Placeholder for property-based tests when exact outputs are hard to compute.
    """
    logging.info("Property-based test generation not yet fully implemented")
    return []