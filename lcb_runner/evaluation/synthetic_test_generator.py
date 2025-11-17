"""
Test Generation Module for RL Self-Repair

Generates synthetic tests from problem specs to drive the RL repair loop without
using ground-truth tests (avoids test leakage).
"""

import json
import re
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple, Iterable
from dataclasses import dataclass

from lcb_runner.lm_styles import LanguageModel, LMStyle  # assumes LMStyle exists


# LLM-generated test categories (5 tests: basic, boundary, edge, adversarial + 1 duplicate)
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
    def __init__(self, model: LanguageModel, base_runner, deterministic_decoding: bool = True, timeout: int = 180):
        self.model = model
        self.base_runner = base_runner
        self.deterministic_decoding = deterministic_decoding  # e.g., temperature=0 inside base_runner if supported
        self.timeout = timeout  # Timeout for self-consistency checks
        self._llm_cache = {}  # Cache for LLM responses: hash(prompt) -> response
        
        # Log which model is being used for test generation
        print(f"  [SyntheticTestGenerator] Initialized with model: {model.model_repr}", file=sys.stderr, flush=True)
        logging.info(f"SyntheticTestGenerator initialized with model: {model.model_repr}")

    def generate_tests(
        self,
        question: str,
        code_type: str = "stdin",   # "stdin" or "call_based"
        num_tests: int = 5,  # LLM generates 5 tests (basic, boundary, edge, adversarial + 1 duplicate)
        self_consistency_checks: int = 3,
        min_confidence: float = 0.5,
        progress_bar=None,
        adversarial_code: str = None  # NEW: Enable adversarial test generation
    ) -> List[TestCase]:
        """
        Generate synthetic test cases for a problem.
        
        If adversarial_code is provided, generates tests targeted at exposing
        bugs in that specific code (no ground-truth leakage).
        """
        if progress_bar:
            progress_bar.set_description("Generating test plan...")
        
        # Log which model is being used
        print(f"  [Test Generation] Using model: {self.model.model_repr}", file=sys.stderr, flush=True)
        logging.info(f"Starting test generation with model: {self.model.model_repr}")
        
        # Log adversarial mode if enabled
        if adversarial_code:
            print(f"  [Test Generation] 🎯 ADVERSARIAL MODE: Analyzing buggy code ({len(adversarial_code)} chars)", file=sys.stderr, flush=True)
            logging.info(f"Adversarial test generation enabled (code length: {len(adversarial_code)})")
        else:
            print(f"  [Test Generation] Standard mode: No buggy code provided", file=sys.stderr, flush=True)
        
        plan_raw = self._generate_test_plan(question, code_type, adversarial_code)
        
        if progress_bar:
            progress_bar.set_description("Generating concrete tests (sequential)...")
        raw_tests = self._generate_concrete_tests_sequential(
            question, plan_raw, num_tests, code_type, adversarial_code, progress_bar
        )
        
        if progress_bar:
            progress_bar.set_description("Validating tests with self-consistency...")
        
        test_cases = self._validate_with_self_consistency(
            question, raw_tests, self_consistency_checks, code_type, min_confidence
        )
        
        # NEW: Add mandatory edge cases to ensure comprehensive coverage
        if progress_bar:
            progress_bar.set_description("Adding mandatory edge cases...")
        
        mandatory_tests = self._generate_mandatory_edge_cases(
            question, code_type, self_consistency_checks, min_confidence
        )
        
        # Merge tests, avoiding duplicates
        test_cases = self._merge_test_cases(test_cases, mandatory_tests)
        
        # NEW: Post-synthesis validation - check category distribution
        category_counts = {}
        for tc in test_cases:
            cat = tc.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"\n  === FINAL TEST DISTRIBUTION ===", file=sys.stderr, flush=True)
        print(f"  Total tests: {len(test_cases)}", file=sys.stderr, flush=True)
        for cat in ['basic', 'boundary', 'edge', 'adversarial']:
            count = category_counts.get(cat, 0)
            status = "✓" if count > 0 else "⚠️  MISSING"
            print(f"  {status} {cat}: {count}", file=sys.stderr, flush=True)
        
        print(f"  === END DISTRIBUTION ===\n", file=sys.stderr, flush=True)
        
        # Log cache statistics
        cache_stats = self.get_cache_stats()
        logging.info(f"LLM Cache: {cache_stats['cache_size']} entries cached")
        
        if progress_bar:
            progress_bar.set_description(f"Validated {len(test_cases)} tests")
            progress_bar.update(1)
        
        return test_cases

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

    def _llm_respond(self, user_content: str, system_content: Optional[str] = None, use_cache: bool = True) -> str:
        """
        Single entrypoint to call the base_runner with caching support.
        Caching is especially useful for self-consistency checks which use identical prompts.
        
        Args:
            user_content: The user prompt
            system_content: Optional system prompt
            use_cache: If True, cache and reuse responses for identical prompts
        """
        # Create cache key from prompt content
        cache_key = hash((user_content, system_content or ""))
        
        # Check cache first
        if use_cache and cache_key in self._llm_cache:
            logging.debug(f"Cache HIT for prompt hash {cache_key}")
            return self._llm_cache[cache_key]
        
        logging.debug(f"Cache MISS for prompt hash {cache_key}")
        
        # Log which model is making the call (for verification)
        logging.debug(f"Making LLM call with model: {self.model.model_repr}")
        
        # Make LLM call
        if self._is_chat_style():
            messages = []
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": user_content})
            # If your base_runner supports per-call params, pass temperature=0 when deterministic
            outputs = self.base_runner.prompts_to_outputs([messages])
        else:
            outputs = self.base_runner.prompts_to_outputs([user_content])
        
        response = outputs[0][0] if outputs and outputs[0] else ""
        
        # Store in cache
        if use_cache:
            self._llm_cache[cache_key] = response
            logging.debug(f"Cached response for hash {cache_key} (cache size: {len(self._llm_cache)})")
        
        return response
    
    def clear_cache(self):
        """Clear the LLM response cache."""
        self._llm_cache.clear()
        logging.info("LLM cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._llm_cache),
            "total_cached_responses": len(self._llm_cache)
        }

    def _generate_test_plan(self, question: str, code_type: str, adversarial_code: str = None) -> str:
        # Build adversarial context if buggy code is provided
        adversarial_context = ""
        if adversarial_code:
            # Limit code size to avoid token overflow
            code_snippet = adversarial_code[:2000] if len(adversarial_code) > 2000 else adversarial_code
            adversarial_context = f"""
🔍 ANALYZE THIS USER-SUBMITTED SOLUTION FOR POTENTIAL WEAKNESSES:

```python
{code_snippet}
```

Your task: Identify potential bugs or missed edge cases in this code.

Consider:
- Does it fail on empty inputs, single elements, or boundary cases?
- Does it handle special values (0, negatives, duplicates) correctly?
- Are there off-by-one errors or incorrect loop conditions?
- Does it have O(n²) complexity that could timeout on large inputs?
- Does it make incorrect assumptions about input structure?
- Are there cases where the logic breaks down (e.g., k=4 vs k=2)?

Based on your analysis, design tests that will EXPOSE these specific weaknesses.
"""
        
        prompt = f"""Given this problem specification, design a test plan that will catch incorrect/inefficient algorithms:

Problem:
{question}

{adversarial_context}

Design a diverse test plan with these categories:
1) Basic cases - Small typical inputs (n ≤ 20)
2) Boundary conditions - Min/max values, empty, single element (n ≤ 10)
3) Edge cases - Special values (0, negative, duplicates, all same, n ≤ 50)
4) Adversarial cases - Tricky inputs that expose {'the specific weaknesses you found' if adversarial_code else 'common bugs'} (n ≤ 50)

Code type: {code_type}

For each category, describe the intent and representative inputs.
Provide the test plan in JSON format with fields: category, intent, example_inputs.
"""
        
        system_prompt = "You are an expert at designing comprehensive test cases by analyzing problem specifications and identifying code weaknesses."
        response = self._llm_respond(
            user_content=prompt,
            system_content=system_prompt
        )
        logging.info(f"Generated test plan (truncated): {response[:200]}...")
        return response

    def _generate_concrete_tests(
        self,
        question: str,
        test_plan: str,
        num_tests: int,
        code_type: str,
        adversarial_code: str = None
    ) -> List[Dict[str, Any]]:
        # Build adversarial context if buggy code is provided
        adversarial_context = ""
        if adversarial_code:
            code_snippet = adversarial_code[:1500] if len(adversarial_code) > 1500 else adversarial_code
            adversarial_context = f"""

🎯 ADVERSARIAL CONTEXT - Analyze this buggy code:
```python
{code_snippet}
```

Your tests should be designed to EXPOSE weaknesses in this specific code.
Identify where the logic might break down and create tests that target those cases.
"""
        
        # Use f-string directly - the doubled braces {{}} will be rendered as single braces in the output
        prompt = f"""Based on this test plan, generate {num_tests} DIVERSE test cases with EXACT expected outputs.

🚨 CRITICAL REQUIREMENTS:

1. LOGICAL REASONING ONLY:
   - Derive outputs step-by-step from problem logic
   - Do NOT reference solutions, hints, or patterns from examples
   - Show reasoning for non-trivial cases

2. MANDATORY DIVERSITY:
   - ALL {num_tests} tests MUST be DIFFERENT inputs
   - NO duplicate or near-duplicate inputs allowed
   - Cover ALL categories: basic, boundary, edge, adversarial
   - Each test must expose a DIFFERENT potential bug

3. ADVERSARIAL FOCUS:
   - Design tests to FAIL incorrect algorithms
   - Include edge cases that naive solutions miss
   - For each test, explain what incorrect approach it catches
   - Example: "This test fails solutions that only check parity"

Problem:
{question}

Test Plan:
{test_plan}
{adversarial_context}

Test Categories (MUST cover ALL):
- basic: Small typical inputs (variety of structures)
- boundary: Min/max values, empty, single element
- edge: Special values (0, negative, duplicates, all same, all unique)
- adversarial: Tricky cases that expose common bugs

⚠️ INPUT FORMATTING:
- NEVER use ellipsis (...) - generate FULL arrays/strings
- For stdin: include ALL lines as they would appear
- For function calls: wrap according to parameter count
  • ONE parameter: [[...]]
  • MULTIPLE parameters: [arg1, arg2, ...]

"""

        prompt += """

For stdin problems, the input should be formatted exactly as it would appear in stdin (including newlines).

For function-based problems, the input should be a valid JSON array or object representing the function parameters.

⚠️ CRITICAL INPUT FORMATTING (prevents TypeErrors):

Step 1: Count function parameters from signature
Step 2: Format accordingly:

If function takes ONE parameter (regardless of type):
  → Wrap in array: [[...]]
  
  Examples:
  • def solve(nums: List[int]) → Input: [[1, 2, 3]]
  • def solve(words: List[str]) → Input: [["hello", "world"]]  ← WRAP IT!
  • def solve(matrix: List[List[int]]) → Input: [[[1, 2], [3, 4]]]
  
If function takes MULTIPLE parameters:
  → Flat array: [arg1, arg2, ...]
  
  Examples:
  • def solve(s: str, k: int) → Input: ["hello", 5]
  • def solve(arr: List[int], target: int) → Input: [[1, 2, 3], 10]
    (note: the list parameter itself is NOT double-wrapped when it's one of multiple params)

COMMON MISTAKE: ["a", "b"] when function takes ONE list param
CORRECT: [["a", "b"]] ← Always wrap if ONE parameter!

Provide as a JSON array with objects containing:
- category: basic/boundary/edge/adversarial
- input: the input value (for stdin problems, include ALL lines including test count)
- expected: the expected output (as string, exactly as it should be printed)
- rationale: brief explanation of what the test validates

Code type: {code_type}

IMPORTANT: Return ONLY a valid JSON array, no markdown, no code blocks, no extra text."""

        # Add code-type-specific examples
        if code_type == "stdin":
            prompt += """
Example format for stdin:
[
  {{"category": "basic", "input": "3\\n1 2 3", "expected": "6", "rationale": "Basic sum"}},
  {{"category": "boundary", "input": "0", "expected": "0", "rationale": "Empty input"}},
  {{"category": "edge", "input": "5\\n1 1 1 1 1", "expected": "5", "rationale": "All same elements"}}
]"""
        else:  # call_based
            prompt += """
Example format for call_based:

CORRECT Examples (ONE parameter functions):
[
  {{"category": "basic", "input": "[[1, 2, 3, 4]]", "expected": "10", "rationale": "ONE list param → wrapped"}},
  {{"category": "basic", "input": "[[\\\"hello\\\", \\\"world\\\"]]", "expected": "2", "rationale": "ONE list of strings → wrapped [[ ]]"}},
  {{"category": "boundary", "input": "[[]]", "expected": "0", "rationale": "Empty list → still wrapped [[ ]]"}},
  {{"category": "adversarial", "input": "[[1, 1, 1, 1, 1]]", "expected": "5", "rationale": "All identical values"}}
]

CORRECT Examples (MULTIPLE parameter functions):
[
  {{"category": "basic", "input": "[\\\"abcac\\\", 2]", "expected": "1", "rationale": "TWO params → flat array"}},
  {{"category": "basic", "input": "[[1, 2, 3], 10]", "expected": "true", "rationale": "TWO params (list, int)"}}
]

WRONG (DO NOT DO THIS):
❌ {{"input": "[\\\"a\\\", \\\"b\\\"]"}}  ← Missing outer wrapper if function takes ONE list param!
✅ {{"input": "[[\\\"a\\\", \\\"b\\\"]]"}}  ← Correct!

**ALWAYS wrap if function signature has ONE parameter!**"""

        prompt = f"{prompt}\n"
        response = self._llm_respond(
            user_content=prompt,
            system_content="You are an expert at computing exact expected outputs for test cases. Return ONLY JSON, no other text."
        )
        
        # Log the raw response for debugging
        logging.info(f"LLM response for test generation (first 500 chars): {response[:500]}")
        
        tests = self._parse_json_array(response)
        if not tests:
            logging.warning(f"Failed to parse JSON, attempting manual extraction...")
            logging.warning(f"Full response (first 1000 chars): {response[:1000]}")  # Additional logging
            tests = self._manual_extract_tests(response)
        
        logging.info(f"Parsed {len(tests)} test objects from LLM response")
            
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
        seen_inputs = set()  # NEW: Track inputs to reject duplicates
        
        for i, t in enumerate(tests):
            cat = str(t.get("category", "basic")).lower()
            if cat not in CATEGORIES:
                cat = "basic"
            if "input" not in t or "expected" not in t:
                logging.warning(f"Test {i} missing input or expected: {t}")
                continue

            # Clean up input and expected values - AGGRESSIVE cleaning
            input_val = t["input"]
            if isinstance(input_val, str):
                # Strip whitespace, then trailing commas, then quotes
                cleaned = input_val.strip().rstrip(',').strip()
                # Remove outer quotes (both single and double)
                while (cleaned.startswith('"') and cleaned.endswith('"')) or \
                      (cleaned.startswith("'") and cleaned.endswith("'")):
                    cleaned = cleaned[1:-1].strip()
                
                # Unescape sequences like \n to actual newlines
                try:
                    input_val = bytes(cleaned, "utf-8").decode("unicode_escape")
                except (UnicodeDecodeError, AttributeError):
                    input_val = cleaned
            
            expected_val = t["expected"]
            if isinstance(expected_val, str):
                cleaned = expected_val.strip().rstrip(',').strip()
                # Remove outer quotes
                while (cleaned.startswith('"') and cleaned.endswith('"')) or \
                      (cleaned.startswith("'") and cleaned.endswith("'")):
                    cleaned = cleaned[1:-1].strip()
                expected_val = cleaned

            # Reject malformed inputs with code-like syntax or ellipsis
            if isinstance(input_val, str):
                code_patterns = [".repeat(", ".join(", ' + "', '") + "', '" + "']
                if any(pattern in input_val for pattern in code_patterns):
                    logging.warning(f"Test {i} rejected: input contains code-like syntax: {repr(input_val[:50])}")
                    continue
                # CRITICAL: Reject inputs with ellipsis (invalid JSON)
                if '...' in input_val:
                    logging.warning(f"Test {i} rejected: input contains ellipsis (invalid JSON): {repr(input_val[:100])}")
                    continue
            
            # NEW: Reject duplicate inputs
            input_normalized = ' '.join(str(input_val).split())  # Normalize whitespace
            if input_normalized in seen_inputs:
                logging.warning(f"Test {i} rejected: DUPLICATE input (already seen)")
                print(f"⚠️  Test {i} DUPLICATE rejected: {str(input_val)[:100]}", file=sys.stderr, flush=True)
                continue
            seen_inputs.add(input_normalized)
            
            # Auto-correct call_based inputs that aren't JSON arrays/objects
            if code_type == "call_based" and isinstance(input_val, str):
                # Try to parse as JSON first
                try:
                    json.loads(input_val)
                except (json.JSONDecodeError, ValueError):
                    # Not valid JSON, try to fix it
                    # If it looks like a plain string value, wrap it in a JSON array
                    if not input_val.startswith('[') and not input_val.startswith('{'):
                        logging.warning(f"Test {i}: Auto-correcting non-JSON input '{input_val}' to '[{json.dumps(input_val)}]'")
                        input_val = json.dumps([input_val])

            valid.append({
                "category": cat,
                "input": input_val,
                "expected": expected_val,
                "rationale": t.get("rationale", "")
            })
        print(f"Generated {len(valid)} concrete tests")
        
        # Validate and auto-fix input format for call_based tests
        if code_type == "call_based" and valid:
            valid = self._validate_and_fix_input_format(valid, question)
        
        if len(valid) > 0:
            # Ensure input is a string before slicing for the log
            input_for_log = str(valid[0]['input'])
            print(f"Sample test - Input: {repr(input_for_log[:100])}, Expected: {repr(valid[0]['expected'])}")
        return valid

    def _generate_concrete_tests_sequential(
        self,
        question: str,
        test_plan: str,
        num_tests: int,
        code_type: str,
        adversarial_code: str = None,
        progress_bar=None
    ) -> List[Dict[str, Any]]:
        """
        Generate tests one by one (sequential) to prevent timeouts.
        Each test is generated and validated individually.
        
        Trade-offs:
        - PRO: Less likely to timeout (smaller requests)
        - PRO: Can see progress incrementally
        - PRO: Can stop early if needed
        - CON: Slower overall (can't parallelize)
        - CON: More API calls (but same total cost)
        """
        raw_tests = []
        
        # Generate tests one at a time
        # Use stderr to avoid conflicts with progress bar (which uses stdout)
        print(f"  Generating {num_tests} tests sequentially (one by one)...", file=sys.stderr, flush=True)
        for i in range(num_tests):
            print(f"\n  [Test {i+1}/{num_tests}] Generating...", file=sys.stderr, flush=True)
            if progress_bar:
                progress_bar.set_description(f"Generating test {i+1}/{num_tests}...")
            
            # NEW: Build context from previously generated tests (prevents duplicates)
            previous_context = ""
            if raw_tests:
                previous_context = "\n\n🚨 PREVIOUSLY GENERATED TESTS - DO NOT DUPLICATE THESE:\n"
                # Show last 3 tests to avoid context overflow
                recent_tests = raw_tests[-3:] if len(raw_tests) > 3 else raw_tests
                for j, prev in enumerate(recent_tests):
                    prev_input = str(prev.get('input', ''))
                    prev_category = prev.get('category', 'unknown')
                    # Truncate long inputs for context
                    if len(prev_input) > 150:
                        prev_input = prev_input[:150] + "..."
                    previous_context += f"  Test {len(raw_tests) - len(recent_tests) + j + 1}: category={prev_category}, input={prev_input}\n"
                
                # NEW: Show which categories are still needed
                categories_used = [t.get('category', 'unknown') for t in raw_tests]
                categories_needed = []
                for cat in ['basic', 'boundary', 'edge', 'adversarial']:
                    if cat not in categories_used:
                        categories_needed.append(cat)
                
                previous_context += f"\n⚠️ Your new test MUST:\n"
                previous_context += f"  1. Have DIFFERENT input values/structure than above\n"
                if categories_needed:
                    previous_context += f"  2. Use a category we haven't covered yet: {', '.join(categories_needed)}\n"
                else:
                    previous_context += f"  2. Can repeat a category, but input MUST be different\n"
            
            try:
                # Determine which category to focus on
                categories_used = [t.get('category', 'unknown') for t in raw_tests]
                categories_needed = [cat for cat in ['basic', 'boundary', 'edge', 'adversarial'] if cat not in categories_used]
                
                if categories_needed:
                    # Prioritize uncovered categories
                    category_focus = f"one of these uncovered categories: {', '.join(categories_needed)}"
                else:
                    # All 4 covered, 5th test can duplicate a category with DIFFERENT input
                    category_focus = "duplicate one of [basic, boundary, edge, adversarial] with DIFFERENT input"
                
                # Build adversarial context if buggy code is provided
                adversarial_hint = ""
                if adversarial_code:
                    code_snippet = adversarial_code[:1500] if len(adversarial_code) > 1500 else adversarial_code
                    adversarial_hint = f"""

🎯 ADVERSARIAL CONTEXT - Analyze this buggy code:
```python
{code_snippet}
```

When generating the test, consider:
- What edge cases might this code miss?
- Where might the logic break down?
- Design your test to EXPOSE these weaknesses
"""
                
                # Generate single test with context
                prompt = f"""Based on this test plan, generate ONE UNIQUE test case with EXACT expected output.

🚨 CRITICAL REQUIREMENTS:

1. LOGICAL REASONING ONLY:
   - Derive output step-by-step from problem logic
   - Do NOT reference solutions or patterns
   - Show reasoning if non-trivial

2. THIS TEST MUST BE DIFFERENT:
   - Input MUST differ from previous {i} test(s)
   - DO NOT repeat the same input values (e.g., [1,1,1,1,1])
   - DO NOT repeat the same CATEGORY unless all 4 categories already covered
   - Test a DIFFERENT edge case or scenario
   - Expose a DIFFERENT potential bug
   - Use DIFFERENT VALUES than previous tests

3. ADVERSARIAL DESIGN:
   - This test should FAIL incorrect algorithms
   - Explain what incorrect approach it catches
   - Example: "Fails solutions that assume sorted input"

4. CATEGORY SELECTION FOR THIS TEST:
   - Focus on: {category_focus}
   - Ensure diversity across test categories

Problem:
{question}

Test Plan:
{test_plan}
{adversarial_hint}

{previous_context}

Test case #{i+1} of {num_tests} - MUST be unique.

❌ WRONG (duplicate values from previous test):
  - If Test 1 used [1,1,1,1,1], do NOT use [1,1,1,1,1] again
  - If Test 2 used "aaaab", do NOT use "aaaab" again

✅ CORRECT (different values):
  - Use [2,3,5,7,11] or [100,200,300] or [-1,0,1]
  - Use "xyzab" or "aaabbbccc" or "abcdefgh"

Category for this test: {category_focus}

ALLOWED Test Categories (ONLY use these):
- basic: Small typical inputs (n ≤ 20, VARIED structures)
- boundary: Min/max values, empty, single element (n ≤ 10)
- edge: Special values, patterns (n ≤ 50)
- adversarial: Tricky cases that expose common bugs

⚠️ INPUT FORMATTING:
- NEVER use Python code (.join(), +, *, etc.) in the "input" field
- Write out FULL arrays/strings directly in JSON
- Keep inputs small (n ≤ 50) for fast testing
- For stdin: include ALL lines as they would appear
- For function calls: wrap according to parameter count
  • ONE parameter: [[...]]
  • MULTIPLE parameters: [arg1, arg2, ...]
"""

                prompt += f"""

For stdin problems, the input should be formatted exactly as it would appear in stdin (including newlines).
For function-based problems, the input should be a valid JSON array or object representing the function parameters.

⚠️ CRITICAL INPUT FORMATTING (prevents TypeErrors):

If function takes ONE parameter (regardless of type):
  → Wrap in array: [[...]]
  
If function takes MULTIPLE parameters:
  → Flat array: [arg1, arg2, ...]

Provide as a JSON object (NOT array) with fields:
- category: basic/boundary/edge/adversarial (ONLY these categories)
- input: the input value
- expected: the expected output (as string)
- rationale: brief explanation

Code type: {code_type}

IMPORTANT: Return ONLY a valid JSON object, no markdown, no code blocks, no extra text.
Example: {{"category": "basic", "input": "[[1, 2, 3]]", "expected": "6", "rationale": "Basic test"}}
"""

                response = self._llm_respond(
                    user_content=prompt,
                    system_content="You are an expert at computing exact expected outputs for test cases. Return ONLY JSON, no other text.",
                    use_cache=False  # Don't cache individual test generation
                )
                
                # Parse single test
                try:
                    import json
                    # Remove markdown code blocks if present
                    cleaned = response.strip()
                    if "```json" in cleaned:
                        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                    elif "```" in cleaned:
                        cleaned = cleaned.split("```")[1].split("```")[0].strip()
                    
                    test_obj = json.loads(cleaned)
                    
                    # Validate required fields
                    if "input" in test_obj and "expected" in test_obj:
                        # Log the generated test
                        input_val = str(test_obj.get("input", ""))
                        expected_val = str(test_obj.get("expected", ""))
                        category = test_obj.get("category", "basic")
                        
                        # Show full test details (use stderr to avoid progress bar conflicts)
                        print(f"    ✓ Test {i+1} generated:", file=sys.stderr, flush=True)
                        print(f"      Category: {category}", file=sys.stderr, flush=True)
                        print(f"      Input: {input_val[:200]}{'...' if len(input_val) > 200 else ''}", file=sys.stderr, flush=True)
                        print(f"      Expected: {expected_val}", file=sys.stderr, flush=True)
                        
                        logging.info(f"Generated test {i+1}/{num_tests}: category={category}, input={input_val[:200]}, expected={expected_val}")
                        
                        raw_tests.append(test_obj)
                    else:
                        logging.warning(f"Test {i+1} missing required fields: {test_obj}")
                except json.JSONDecodeError as e:
                    print(f"    ✗ Test {i+1} failed: JSON parse error", file=sys.stderr, flush=True)
                    logging.warning(f"Failed to parse test {i+1} as JSON: {e}")
                    logging.warning(f"Response (first 200 chars): {response[:200]}")
                    
            except Exception as e:
                print(f"    ✗ Test {i+1} failed: {str(e)[:100]}", file=sys.stderr, flush=True)
                logging.warning(f"Error generating test {i+1}: {e}")
                continue
        
        # Post-process: validate and clean (same as batch version)
        valid = []
        seen_inputs = set()  # Track inputs to reject duplicates
        
        for i, t in enumerate(raw_tests):
            cat = str(t.get("category", "basic")).lower()
            if cat not in CATEGORIES:
                cat = "basic"
            
            input_val = t["input"]
            if isinstance(input_val, str):
                cleaned = input_val.strip().rstrip(',').strip()
                while (cleaned.startswith('"') and cleaned.endswith('"')) or \
                      (cleaned.startswith("'") and cleaned.endswith("'")):
                    cleaned = cleaned[1:-1].strip()
                try:
                    input_val = bytes(cleaned, "utf-8").decode("unicode_escape")
                except (UnicodeDecodeError, AttributeError):
                    input_val = cleaned
            
            expected_val = t["expected"]
            if isinstance(expected_val, str):
                cleaned = expected_val.strip().rstrip(',').strip()
                while (cleaned.startswith('"') and cleaned.endswith('"')) or \
                      (cleaned.startswith("'") and cleaned.endswith("'")):
                    cleaned = cleaned[1:-1].strip()
                expected_val = cleaned
            
            # Reject ellipsis
            if isinstance(input_val, str) and '...' in input_val:
                logging.warning(f"Test {i+1} rejected: input contains ellipsis")
                continue
            
            # NEW: Reject Python code patterns in JSON (e.g., string concatenation, .join())
            if isinstance(input_val, str):
                python_code_patterns = [' + "', '" + ', '.join(', '.repeat(', '" * ', ' * "']
                if any(pattern in input_val for pattern in python_code_patterns):
                    logging.warning(f"Test {i+1} rejected: input contains Python code (not valid JSON)")
                    print(f"    ⚠️  Test {i+1} rejected: Python code in JSON: {str(input_val)[:150]}", file=sys.stderr, flush=True)
                    continue
            
            # NEW: Reject duplicate inputs
            input_normalized = ' '.join(str(input_val).split())  # Normalize whitespace
            if input_normalized in seen_inputs:
                logging.warning(f"Test {i+1} rejected: DUPLICATE input (already seen)")
                print(f"    \u26a0\ufe0f  Test {i+1} DUPLICATE rejected: {str(input_val)[:100]}", file=sys.stderr, flush=True)
                continue
            seen_inputs.add(input_normalized)
            
            valid.append({
                "category": cat,
                "input": input_val,
                "expected": expected_val,
                "rationale": t.get("rationale", "")
            })
        
        # Validate and auto-fix input format for call_based tests
        if code_type == "call_based" and valid:
            valid = self._validate_and_fix_input_format(valid, question)
        
        # NEW: Validate category diversity
        categories_covered = {t['category'] for t in valid}
        required_categories = {'basic', 'boundary', 'edge', 'adversarial'}
        missing_categories = required_categories - categories_covered
        
        print(f"\n  === TEST GENERATION SUMMARY ===", file=sys.stderr, flush=True)
        print(f"  ✓ Generated {len(valid)}/{num_tests} valid tests", file=sys.stderr, flush=True)
        print(f"  ✓ Categories covered: {', '.join(sorted(categories_covered))}", file=sys.stderr, flush=True)
        
        if missing_categories:
            print(f"  ⚠️  WARNING: Missing categories: {', '.join(sorted(missing_categories))}", file=sys.stderr, flush=True)
            logging.warning(f"Test generation missing categories: {missing_categories}")
        
        if len(valid) > 0:
            print(f"  First test - Input: {repr(str(valid[0]['input'])[:100])}, Expected: {repr(valid[0]['expected'])}", file=sys.stderr, flush=True)
        print(f"  === END TEST GENERATION ===\n", file=sys.stderr, flush=True)
        
        return valid

    def _validate_and_fix_input_format(self, tests: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """
        Auto-fix input format for call_based tests.
        Detects if function takes ONE parameter and wraps input if needed.
        """
        import json
        import re
        
        # Try to extract function signature from problem
        # Look for patterns like: def funcName(param: Type)
        func_match = re.search(r'def\s+\w+\s*\(\s*self\s*,\s*([^)]+)\)', question)
        if not func_match:
            # No clear signature, skip validation
            return tests
        
        params_str = func_match.group(1)
        # Count parameters (split by comma, but be careful of nested types)
        # Simple heuristic: count commas at depth 0
        depth = 0
        param_count = 1  # At least one if we matched
        for char in params_str:
            if char in '[({<':
                depth += 1
            elif char in '])}>':
                depth -= 1
            elif char == ',' and depth == 0:
                param_count += 1
        
        fixed_tests = []
        for test in tests:
            input_val = test['input']
            
            try:
                # Parse input to check structure
                if isinstance(input_val, str):
                    parsed = json.loads(input_val)
                else:
                    parsed = input_val
                
                # If function takes ONE parameter and input is a list (not wrapped)
                if param_count == 1 and isinstance(parsed, list):
                    # Check if it's already wrapped (first element is also a list)
                    # But be careful: [["a", "b"]] vs ["a", "b"]
                    # The issue: if we have ["str1", "str2"], we need [["str1", "str2"]]
                    
                    # Heuristic: If all elements are strings/numbers/dicts (not lists),
                    # it's likely unwrapped
                    needs_wrapping = all(not isinstance(item, list) for item in parsed)
                    
                    if needs_wrapping:
                        # Wrap it
                        fixed_input = [parsed]
                        test['input'] = json.dumps(fixed_input)
                        print(f"  ⚙️  Auto-fixed input format: {input_val[:50]} → {json.dumps(fixed_input)[:50]}")
                
                fixed_tests.append(test)
                
            except (json.JSONDecodeError, TypeError, AttributeError):
                # Can't parse, keep original
                fixed_tests.append(test)
        
        return fixed_tests

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
            print(f"Test {idx}: confidence={confidence:.2f}, baseline='{baseline}', original_expected='{test.get('expected', 'N/A')}'")
            if confidence >= min_confidence:
                validated.append(TestCase(
                    input_val=test["input"],
                    expected_output=baseline,
                    confidence=confidence,
                    kind="exact",
                    category=test.get("category", "basic"),
                    rationale=test.get("rationale", "")
                ))
            else:
                print(f"Test {idx} rejected: confidence {confidence:.2f} < {min_confidence} (self-consistency checks disagreed)")
        
        if len(validated) == 0 and len(raw_tests) > 0:
            print(f"⚠️  All {len(raw_tests)} tests rejected due to timeouts or low confidence")
            print(f"    This often means the reference solution is too slow or has issues")
        else:
            print(f"Validated {len(validated)}/{len(raw_tests)} tests")
        return validated

    def _compute_expected_with_consistency(
        self,
        question: str,
        input_val: Any,
        num_checks: int,
        code_type: str
    ) -> Tuple[Any, float]:
        import threading
        
        outputs: List[str] = []
        
        # NEW: Smart truncation for large inputs to avoid token overflow
        input_str = json.dumps(input_val)
        if len(input_str) > 10000:  # If input is too large
            # For large arrays, show pattern instead of full array
            try:
                parsed = json.loads(input_str)
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], list) and len(parsed[0]) > 100:
                        # Large array case
                        arr = parsed[0]
                        n = len(arr)
                        sample = arr[:50] + arr[-50:]
                        input_display = f"[[array of {n} elements: first 50={arr[:50]}, last 50={arr[-50:]}]]"
                    else:
                        input_display = input_str[:10000] + "...[truncated]"
                else:
                    input_display = input_str[:10000] + "...[truncated]"
            except:
                input_display = input_str[:10000] + "...[truncated]"
        else:
            input_display = input_str
        
        for check_idx in range(max(1, num_checks)):
            prompt = f"""Given this problem:
{question}

And this test input:
{input_display}

Compute the expected output exactly.
Code type: {code_type}

IMPORTANT: Your response must ONLY contain the final answer value, nothing else. No explanations, no reasoning, no extra text.
Just output the single expected value (e.g., "YES", "42", "First", etc.).
"""
            
            # Use threading to implement timeout (works on Windows)
            result = [None]
            exception = [None]
            
            def llm_call():
                try:
                    result[0] = self._llm_respond(
                        user_content=prompt,
                        system_content="You are an expert at computing exact expected outputs."
                    )
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=llm_call)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.timeout)
            
            if thread.is_alive():
                logging.warning(f"Self-consistency check {check_idx+1}/{num_checks} timed out after {self.timeout}s (likely testing slow code)")
                outputs.append("")  # Empty output = no agreement
            elif exception[0]:
                logging.warning(f"Self-consistency check {check_idx+1}/{num_checks} failed: {exception[0]}")
                outputs.append("")
            elif result[0]:
                outputs.append(self._strip_fences(result[0]).strip())
            else:
                outputs.append("")

        # Extract final answer if LLM included reasoning
        # Try enhanced extraction first, fallback to original if needed
        baseline = self._extract_answer_enhanced(outputs[0])
        if not baseline or baseline == outputs[0]:  # If enhanced didn't help, try original
            baseline = self._extract_final_answer(outputs[0])
        
        # If baseline is empty, it means all checks failed/timed out - return 0 confidence
        if not baseline or baseline == "":
            return "", 0.0
        
        other_extracted = []
        for o in outputs[1:]:
            extracted = self._extract_answer_enhanced(o)
            if not extracted or extracted == o:
                extracted = self._extract_final_answer(o)
            other_extracted.append(extracted)
        
        agree = sum(1 for o in other_extracted if self._outputs_match(baseline, o))
        confidence = (agree + 1) / max(1, num_checks)
        return baseline, confidence
    
    def _extract_answer_enhanced(self, text: str) -> str:
        """
        Enhanced answer extraction that handles non-numeric answers (YES/NO, true/false, strings).
        Returns extracted answer or original text if no extraction pattern matches.
        """
        text = text.strip()
        if not text:
            return text
        
        # Pattern 1: Look for "answer is/was X" or "result is X" or "output is: X" type patterns
        # These patterns allow the answer to be on the same line or next line after a colon
        answer_patterns = [
            # Match "output/answer is:" followed by answer on next line (after blank lines)
            r'(?:expected\s*)?(?:output|answer|result)\s+.*?\s+is:\s*\n+\s*(\w+)',
            r'(?:final\s*answer|answer|result|output|expected)\s*(?:is|was|would\s*be)\s*[:\-]?\s*\n?\s*[\'"]?([^\n\'"\.]+?)[\'"]?\s*(?:\.|$|\n)',
            r'(?:thus|therefore|so),?\s*(?:the\s*)?(?:answer|output|result)\s*(?:is|would\s*be|was)\s*[:\-]?\s*\n?\s*[\'"]?([^\n\'"\.]+?)[\'"]?\s*(?:\.|$|\n)',
            r'(?:expected\s*)?output\s*is\s*[:\-]?\s*\n?\s*[\'"]?([^\n\'"\.]+?)[\'"]?\s*(?:\.|$|\n)',
            r'(?:the\s*)?output\s+.*?\s+is\s+[\'"]([^\'"]+)[\'"]',  # "the output ... is 'X'"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Clean up common artifacts
                extracted = re.sub(r'\s*\.$', '', extracted)
                return extracted
        
        # Pattern 2: Look for standalone values on last non-empty lines
        # CRITICAL: Handle multi-line outputs (e.g., "2\n1\n2" for stdin problems with multiple test cases)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            # Pattern to match simple answer lines (numbers, YES/NO, true/false, etc.)
            answer_line_pattern = r'^(?:\d+\.?\d*|YES|NO|True|False|true|false|null|\[.*\]|\{.*\}|"[^"]*")$'
            
            # Find consecutive answer lines from the end
            answer_lines = []
            for line in reversed(lines):
                if re.match(answer_line_pattern, line):
                    answer_lines.insert(0, line)  # Build in correct order
                else:
                    break  # Stop at first non-answer line
            
            # If we found answer lines, return them
            if answer_lines:
                # For multi-line outputs (e.g., stdin with multiple test cases), preserve all lines
                if len(answer_lines) > 1:
                    # Check if all are numeric (common for stdin multi-case problems)
                    all_numeric = all(re.match(r'^\d+\.?\d*$', line) for line in answer_lines)
                    if all_numeric:
                        return '\n'.join(answer_lines)  # Preserve multi-line structure
                
                # Default: return as is (single line or multi-line)
                if len(answer_lines) == 1:
                    return answer_lines[0]
                else:
                    return '\n'.join(answer_lines)
        
        # No pattern matched, return original
        return text
    
    def _extract_final_answer(self, text: str) -> str:
        """
        Extract the final numeric answer from LLM output that may include reasoning.
        Tries multiple heuristics to find the actual answer.
        """
        text = text.strip()
        
        # Try to find patterns like "Final answer: 2", "Answer: 2", "result is 2", etc.
        # Look for numeric patterns after common keywords
        patterns = [
            r'(?:final\s*answer|answer|result)\s*[:\-]?\s*(\d+)',
            r'(?:^|\n)\s*(\d+)\s*(?:\n|$)',
            r'=?\s*(\d+)\s*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no pattern found, try to extract the last standalone number on its own line
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            # Look for lines that are mostly just a number
            if re.match(r'^\d+\.?\s*$', line):
                return line.strip('.')
        
        # Last resort: return the whole text (may have reasoning, but that's what we got)
        return text

    # ---------- Mandatory Edge Case Generation ----------

    def _generate_programmatic_large_test(
        self,
        question: str,
        code_type: str,
        self_consistency_checks: int,
        min_confidence: float
    ) -> Optional[TestCase]:
        """
        DEPRECATED: This function is no longer used.
        Previously attempted to generate large tests programmatically,
        but was removed in favor of LLM-only test generation.
        """
        # Function deprecated - no longer generating programmatic tests
        logging.debug("_generate_programmatic_large_test called but is deprecated")
        return None

    def _generate_mandatory_edge_cases(
        self,
        question: str,
        code_type: str,
        self_consistency_checks: int,
        min_confidence: float
    ) -> List[TestCase]:
        """
        Generate mandatory edge cases that should ALWAYS be tested for algorithmic problems.
        These are standard edge cases (not ground-truth leakage) that expose common bugs.
        
        For array/list problems, this includes:
        - Single element
        - Empty array (if applicable)
        - All same elements
        - Monotonic sequences
        
        Expected outputs are computed from the problem description (no ground-truth).
        """
        mandatory_edge_cases = []
        
        # Detect if this is an array/list problem
        is_array_problem = any(keyword in question.lower() for keyword in 
                              ["array", "list", "subarray", "sequence", "nums"])
        
        if not is_array_problem:
            # For non-array problems, don't add mandatory tests (problem-specific)
            return []
        
        # Define mandatory test inputs based on code_type
        if code_type == "call_based":
            edge_case_templates = [
                {
                    "input": "[[1]]",
                    "rationale": "Single element array - tests base case handling"
                },
                {
                    "input": "[[1, 1, 1, 1]]",
                    "rationale": "All identical elements - tests handling of uniform data"
                },
                {
                    "input": "[[1, 2, 3, 4, 5]]",
                    "rationale": "Strictly increasing sequence - tests monotonic pattern"
                },
                {
                    "input": "[[5, 4, 3, 2, 1]]",
                    "rationale": "Strictly decreasing sequence - tests reverse monotonic pattern"
                },
            ]
        else:  # stdin
            # For stdin problems, edge cases are more problem-specific
            # We'll still try some basic patterns but be more conservative
            edge_case_templates = [
                {
                    "input": "1\n1",
                    "rationale": "Single test case with minimal input"
                },
            ]
        
        # For each template, compute expected output using LLM + problem description
        for template in edge_case_templates:
            try:
                # Use self-consistency to compute expected output
                expected, confidence = self._compute_expected_output_with_reasoning(
                    question=question,
                    test_input=template["input"],
                    rationale=template["rationale"],
                    code_type=code_type,
                    num_checks=self_consistency_checks
                )
                
                # Only include if confidence is high enough
                if confidence >= min_confidence:
                    test_case = TestCase(
                        input_val=template["input"],
                        expected_output=expected,
                        confidence=confidence,
                        kind="exact",
                        category="edge",
                        rationale=template["rationale"]
                    )
                    mandatory_edge_cases.append(test_case)
                    logging.info(f"Added mandatory edge case: {template['input'][:50]}... (confidence={confidence:.2f})")
                else:
                    logging.warning(f"Skipped low-confidence mandatory edge case: {template['input'][:50]}... (confidence={confidence:.2f})")
            except Exception as e:
                logging.warning(f"Failed to generate mandatory edge case for {template['input'][:50]}...: {e}")
                continue
        
        return mandatory_edge_cases

    def _compute_expected_output_with_reasoning(
        self,
        question: str,
        test_input: str,
        rationale: str,
        code_type: str,
        num_checks: int = 3
    ) -> tuple[str, float]:
        """
        Compute expected output for a test input by carefully analyzing the problem.
        Uses self-consistency checks to ensure accuracy.
        
        This is NOT ground-truth leakage - we're computing the answer from the problem
        description, just like a human solver would.
        """
        prompt = f"""Problem Description:
{question}

Test Input: {test_input}

Rationale: {rationale}

Task: Carefully read the problem description and determine the EXACT expected output for this input.

Important considerations:
1. Check if the problem mentions special handling for edge cases (single elements, empty inputs, etc.)
2. Look for constraints like "if no valid answer, return -1" or similar
3. Consider what counts as a "valid" solution according to the problem definition
4. For problems about subarrays/subsequences, check if single elements count
5. Pay attention to the return type (integer, string, boolean, etc.)

Think step-by-step:
Step 1: What is the input?
Step 2: According to the problem description, what is being asked?
Step 3: Are there any special cases mentioned for this type of input?
Step 4: What is the correct output?

Output ONLY the final expected value. No explanations, no reasoning text, just the answer.
Examples of correct format:
- For number: 42
- For string: "YES"
- For negative: -1
- For boolean: true

Expected Output (value only):"""
        
        # Use self-consistency for higher accuracy
        outputs = []
        for check_idx in range(num_checks):
            try:
                response = self._llm_respond(
                    user_content=prompt,
                    system_content="You are an expert at analyzing algorithm problems and computing exact expected outputs.",
                    use_cache=(check_idx == 0)  # Cache first call, vary others
                )
                extracted = self._extract_answer_enhanced(response)
                if not extracted:
                    extracted = self._extract_final_answer(response)
                outputs.append(extracted.strip())
            except Exception as e:
                logging.warning(f"Check {check_idx+1}/{num_checks} failed for mandatory edge case: {e}")
                outputs.append("")
        
        # Compute confidence based on agreement
        if not outputs or not outputs[0]:
            return "", 0.0
        
        baseline = outputs[0]
        agree = sum(1 for o in outputs[1:] if self._outputs_match(baseline, o))
        confidence = (agree + 1) / max(1, num_checks)
        
        return baseline, confidence

    def _merge_test_cases(
        self,
        existing_tests: List[TestCase],
        new_tests: List[TestCase]
    ) -> List[TestCase]:
        """
        Merge new tests into existing tests, avoiding duplicates based on input similarity.
        """
        merged = list(existing_tests)
        
        for new_test in new_tests:
            # Check if this test input already exists (approximately)
            is_duplicate = False
            for existing_test in merged:
                if self._test_inputs_similar(existing_test.input_val, new_test.input_val):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(new_test)
                logging.info(f"Added mandatory edge case test: {str(new_test.input_val)[:50]}...")
            else:
                logging.debug(f"Skipped duplicate edge case: {str(new_test.input_val)[:50]}...")
        
        return merged

    def _test_inputs_similar(self, input1: Any, input2: Any) -> bool:
        """
        Check if two test inputs are similar (to avoid duplicates).
        Simple string comparison for now - could be enhanced with semantic similarity.
        """
        # Convert to string first (handles lists, dicts, etc.)
        str1 = str(input1)
        str2 = str(input2)
        
        # Normalize whitespace and compare
        norm1 = ' '.join(str1.split())
        norm2 = ' '.join(str2.split())
        return norm1 == norm2

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
        
        # Try to fix common issues - trailing commas
        fixed = re.sub(r',\s*}', '}', candidate)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        try:
            obj = json.loads(fixed)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

        # Last resort: try demjson3 for lenient parsing
        try:
            import demjson3
            obj = demjson3.decode(candidate)
            if isinstance(obj, dict) and "tests" in obj and isinstance(obj["tests"], list):
                return obj["tests"]
            if isinstance(obj, list):
                return obj
        except Exception as e:
            logging.debug(f"demjson3 parsing failed: {e}")

        # ADDITIONAL parsing attempts (added on top of existing logic):
        
        # Try extracting with balanced bracket matching
        def find_balanced_json_array(s: str) -> str:
            """Find first complete JSON array with properly balanced brackets."""
            start_idx = s.find('[')
            if start_idx == -1:
                return None
            
            depth = 0
            in_string = False
            escape = False
            
            for i in range(start_idx, len(s)):
                char = s[i]
                
                if escape:
                    escape = False
                    continue
                    
                if char == '\\':
                    escape = True
                    continue
                    
                if char == '"' and not escape:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0:
                            return s[start_idx:i+1]
            return None
        
        balanced_json = find_balanced_json_array(candidate)
        if balanced_json:
            try:
                obj = json.loads(balanced_json)
                if isinstance(obj, list):
                    return obj
            except json.JSONDecodeError:
                pass
        
        # Try fixing more JSON issues: single quotes, unquoted keys
        try:
            # Replace single quotes with double quotes (naive approach)
            fixed_quotes = candidate.replace("'", '"')
            obj = json.loads(fixed_quotes)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
        
        # Try to extract individual test objects if array parsing fails
        test_objects = []
        test_pattern = r'\{[^{}]*"category"[^{}]*"input"[^{}]*"expected"[^{}]*\}'
        matches = re.finditer(test_pattern, candidate, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    test_objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        if test_objects:
            logging.info(f"Extracted {len(test_objects)} test objects using regex pattern matching")
            return test_objects

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