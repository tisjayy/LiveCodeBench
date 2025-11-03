# Framework Alignment with Research Goals

## ✅ Fully Implemented

### 1. Strong Signal of Novelty
- **Policy learns test-aware repair strategies**: The RL agent selects from 5 prompt strategies (default, step-by-step reasoning, error-specific feedback, syntax check, edge case check) based on semantic state embeddings.
- **Synthetic test generation**: Uses LLM to generate test plans and concrete I/O pairs with self-consistency validation.
- **Evaluation on unseen hidden tests**: Ground-truth tests used ONLY for final evaluation, not during repair loop.
- **Policy vs heuristic comparison**: Can compare `--use_rl` vs baseline `BaseRunner.run_main_repair`.

### 2. Reward Shaped by Test Coverage
**Status**: ✅ **Implemented**
- Reward weighted by test confidence scores from self-consistency
- Improvement-based reward: `improvement = current_score - old_score`
- Penalty for regression: `-0.5`
- Tracks test categories: `basic`, `boundary`, `edge`, `adversarial`

### 3. Ablations: RL vs No-RL
**Status**: ✅ **Implemented**
- Toggle with `--use_rl` flag
- Non-RL baseline uses ground-truth tests for repair (traditional approach)
- RL version uses synthetic tests for internal feedback

## ⚠️ Partially Implemented or Missing

### 4. Test Coverage / Difficulty Weighting
**Status**: ⚠️ **Partial**
- Tests categorized by difficulty/type (`basic`, `boundary`, `edge`, `adversarial`)
- Reward already weighted by confidence scores
- **Missing**: Explicit weights for different categories (e.g., adversarial tests worth 2x more)

**Recommended Addition**:
```python
CATEGORY_WEIGHTS = {
    "basic": 1.0,
    "boundary": 1.5,
    "edge": 2.0,
    "adversarial": 2.5
}
```

### 5. Curriculum Learning (Easy → Hard)
**Status**: ❌ **Not Implemented**
- Benchmark currently sorted by `question_id` (line 68 in scenario_router.py)
- Problems have `difficulty` field (`easy`, `medium`, `hard`)
- No curriculum ordering implemented

**Recommended Addition** in `scenario_router.py`:
```python
# After line 67 in scenario_router.py
if args.curriculum_learning and scenario == Scenario.selfrepair:
    benchmark = sorted(benchmark, key=lambda x: (
        0 if x.difficulty == "easy" else 
        1 if x.difficulty == "medium" else 2
    ))
```

### 6. Random Policy Baseline
**Status**: ❌ **Not Implemented**
- Only two modes: RL policy learning vs non-RL heuristic
- No random action selection baseline

**Recommended Addition** in `rl_agent.py`:
```python
def select_action(self, state: dict, random_policy=False):
    if random_policy:
        return random.randint(0, self.action_dim - 1)
    # ... existing policy logic
```

## Summary

**Fully Aligned Goals**: 3/6 ✅
- Test-aware policy learning
- Synthetic test generation
- Hidden test evaluation
- RL vs Heuristic comparison

**Partially Aligned**: 1/6 ⚠️
- Test coverage/difficulty weighting (categories exist, weights missing)

**Not Implemented**: 2/6 ❌
- Curriculum learning (easy→hard ordering)
- Random policy baseline

## Priority Recommendations

**High Priority** (for paper):
1. Add random policy baseline for ablation
2. Implement curriculum learning ordering
3. Add explicit category weights to reward function

**Low Priority** (nice to have):
4. Add difficulty-based weights to test generation
5. Add adaptive curriculum based on model performance
6. Add more sophisticated reward shaping (e.g., diversity bonus)

