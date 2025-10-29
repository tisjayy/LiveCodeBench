# RL Self-Repair Test Leakage Fix

## Problem Identified

The RL loop was "peeking" at the dataset's provided tests to guide repairs, causing **test leakage**. The reward signal was computed from the official ground-truth tests, meaning the policy was being optimized directly against the evaluation set. This violates a clean separation between training/tuning and evaluation, and can overstate performance.

## Solution Implemented

### Revised Pipeline

1. **Base Codegen + Initial Candidate**: Generate initial code from problem spec (unchanged)

2. **Test Generation (Internal Oracle)**:
   - Use the LLM to produce a test plan covering normal, boundary, adversarial, and randomized cases
   - Generate concrete test I/O pairs with self-consistency checks (multiple samples must agree)
   - Attach confidence scores to each test based on agreement rate
   - Only validated tests with high confidence are kept

3. **RL Self-Repair Loop**:
   - State includes `synthetic_test_summary` with test metadata
   - Policy network chooses a strategy and builds the repair prompt
   - Evaluate repaired code **only on synthetic tests** (not ground-truth tests)
   - Compute reward from pass deltas on synthetic tests with confidence weighting
   - Iterate until pass or max attempts
   - **No ground-truth tests are used in this loop**

4. **Final Evaluation Only**:
   - Run the final code on the dataset's provided tests to report accuracy
   - Log comparison between synthetic test results and ground-truth results

### Files Created/Modified

#### New Files

1. **`lcb_runner/evaluation/synthetic_test_generator.py`**
   - `SyntheticTestGenerator` class that generates tests from problem specifications
   - Three-stage generation: test plan → concrete tests → self-consistency validation
   - Confidence scoring based on agreement across multiple LLM samples
   - Support for both stdin and call-based code types

2. **`lcb_runner/evaluation/internal_evaluator.py`**
   - `InternalEvaluator` class that evaluates code against synthetic tests
   - Separate evaluation paths for stdin-based and call-based code
   - Returns pass/fail results and metadata
   - `compute_weighted_reward()` function that weights rewards by test confidence

#### Modified Files

1. **`lcb_runner/runner/rl_self_repair_runner.py`**
   - Added imports for synthetic test generation and internal evaluation
   - Initialize `SyntheticTestGenerator` and `InternalEvaluator` instances
   - `run_sample_episode()` method completely rewritten:
     - Step 1: Generate synthetic tests from problem spec
     - Step 2: Evaluate initial code on synthetic tests
     - Step 3: RL loop uses synthetic tests for reward (not ground-truth)
     - Step 4: Final evaluation on ground-truth tests only at the end
   - State representation updated to include internal test diagnostics
   - Added `_determine_code_type()` helper method
   - Removed old `compute_reward_from_test_cases()` method (replaced by `compute_weighted_reward`)

### Key Changes

#### Before (Test Leakage)
```python
# Inside RL loop - WRONG!
eval_args = ([repaired_code], sample.get_evaluation_sample(), ...)
new_eval_results, _ = evaluate_generations_by_problem(eval_args)
new_test_results = new_eval_results[0]  # Ground-truth tests!
reward = self.compute_reward_from_test_cases(new_test_results, ...)
```

#### After (No Leakage)
```python
# Step 1: Generate synthetic tests once
synthetic_tests = self.test_generator.generate_tests(question=..., code_type=...)

# Inside RL loop - CORRECT!
new_synth_results, _ = self.internal_evaluator.evaluate_on_synthetic_tests(
    code=repaired_code, tests=synthetic_tests, ...
)
reward = compute_weighted_reward(
    test_results=new_synth_results, test_confidences=[...], ...
)

# Step 5: Final evaluation on ground-truth tests ONLY at the end
final_gt_results, _ = evaluate_generations_by_problem([final_code], ...)
```

### State Representation

The state now includes internal test diagnostics:
```python
state = {
    "question": sample.question_content, 
    "code": current_code, 
    "metadata": current_metadata,
    "synthetic_test_summary": {
        "total": len(synthetic_tests),
        "categories": list(set(t.category for t in synthetic_tests))
    }
}
```

### Test Generation Prompts

The implementation uses three types of prompts:

1. **Test Plan Prompt**: Asks the LLM to create a test plan with categories (basic, boundary, edge, adversarial)
2. **Concrete Test Prompt**: Asks the LLM to instantiate concrete I/O pairs with exact expected outputs
3. **Self-Consistency Prompt**: Asks the LLM to compute the same expected output multiple times independently

Agreement across multiple samples provides confidence scores.

### Reward Function

The new `compute_weighted_reward()` function:
- Returns 1.0 if all tests pass
- Computes weighted pass rate using test confidence scores
- Compares against previous results to reward improvement or penalize regression
- Returns values in range [-0.5, 1.0]

### What to Report in Paper

1. **Clear Separation Diagram**: Show internal test generation + RL repair loop (training) vs. final evaluation on ground-truth tests (held-out)

2. **Ablations**:
   - RL with synthetic tests vs. round-robin strategies vs. random strategies
   - With vs. without property-based tests (when implemented)
   - Confidence-weighted rewards vs. unweighted
   - Different numbers of synthetic tests

3. **Reliability of Test Oracle**:
   - Agreement rates between synthetic tests and dataset tests for overlapping cases
   - Self-consistency agreement percentages for expected outputs
   - Coverage analysis: do synthetic tests exercise the same code paths?

4. **Performance Impact**:
   - Pass rates on ground-truth tests (the only metric that matters)
   - Efficiency: additional LLM calls for test generation (but only once per problem)

### Next Steps / Future Work

1. Implement property-based tests for cases where exact outputs are hard to compute
2. Add more sophisticated confidence scoring (e.g., based on problem type, code complexity)
3. Consider test regeneration or pruning based on low agreement
4. Add ablation studies comparing different numbers of synthetic tests
5. Analyze the correlation between synthetic test pass rates and ground-truth pass rates

## Testing

To verify the fix works:
1. Run the RL self-repair scenario on a small subset of problems
2. Check logs to confirm synthetic tests are generated
3. Verify that ground-truth tests are only used in final evaluation
4. Compare metrics before/after the fix (should show more realistic performance)

