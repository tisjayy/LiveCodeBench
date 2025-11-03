# Two-Stage Repair System

## Overview

The RL self-repair runner now supports a **two-stage repair strategy** that optimizes repair based on problem difficulty:

```
Problem with pass@1 < 1.0
    ↓
Check pass@1 threshold (default: 0.7)
    ↓
    ├─ Yes (pass@1 > 0.7) → Basic Self-Repair
    │   └─ Quick syntax fixes, simple corrections
    │   └─ No RL training, just prompt-based repair
    │   └─ Fast, suitable for minor bugs
    │
    └─ No (pass@1 ≤ 0.7) → RL Self-Repair
        └─ Generate synthetic tests
        └─ RL agent selects strategies
        └─ Multiple repair iterations with learning
        └─ Fallback: If basic repair fails, use RL
```

## Key Features

### 1. **Basic Repair (Fast Path)**
- Used when `pass@1 > threshold` (default: 0.7)
- Single prompt-based repair attempt
- No synthetic test generation
- No RL training overhead
- Suitable for:
  - High pass rate problems (9/10 or 8/10 passed)
  - Minor syntax errors
  - Simple logic mistakes

### 2. **RL Repair (Deep Path)**
- Used when `pass@1 ≤ threshold` (default: 0.7)
- Full RL-based repair with synthetic tests
- Multiple repair iterations (up to 5)
- RL agent learns optimal strategies
- Suitable for:
  - Low pass rate problems (≤7/10 passed)
  - Complex algorithmic errors
  - Edge case handling issues

### 3. **Fallback Mechanism**
- If basic repair fails to produce valid code
- Automatically falls back to RL repair
- Ensures robust repair for all problems

## Usage

### Basic Usage (Default)
```bash
python -m lcb_runner.runner.main \
  --model deepseek-coder \
  --scenario selfrepair \
  --codegen_n 10 \
  --temperature 0.2 \
  --use_rl \
  --start_date 2023-09-01 \
  --end_date 2023-09-30
```

### Disable Basic Repair (Always Use RL)
```bash
python -m lcb_runner.runner.main \
  --model deepseek-coder \
  --scenario selfrepair \
  --use_rl \
  --no_basic_repair
```

### Adjust Threshold
```bash
# More aggressive basic repair (use for pass@1 > 0.5)
python -m lcb_runner.runner.main \
  --model deepseek-coder \
  --scenario selfrepair \
  --use_rl \
  --basic_repair_threshold 0.5

# More conservative (only use basic repair for pass@1 > 0.9)
python -m lcb_runner.runner.main \
  --model deepseek-coder \
  --scenario selfrepair \
  --use_rl \
  --basic_repair_threshold 0.9
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable_basic_repair` | flag | True | Enable basic repair for high-pass-rate problems |
| `--no_basic_repair` | flag | - | Disable basic repair, always use RL |
| `--basic_repair_threshold` | float | 0.7 | Pass@1 threshold for basic vs RL repair |

## Expected Performance

### Basic Repair
- **Speed**: ~10-30 seconds per problem
- **Success Rate**: Good for pass@1 > 0.7
- **Resource Usage**: Minimal (1 LLM call)

### RL Repair
- **Speed**: ~3-10 minutes per problem
- **Success Rate**: Better for difficult problems
- **Resource Usage**: Higher (synthetic tests + multiple iterations)

## Examples

### Problem: pass@1 = 0.9 (9/10 passed)
```
→ Basic Repair triggered
→ Single repair attempt
→ If successful: Return repaired code
→ If failed: Fall back to RL repair
```

### Problem: pass@1 = 0.3 (3/10 passed)
```
→ RL Repair triggered directly
→ Generate synthetic tests
→ Run 5 repair iterations with RL agent
→ Track best attempt
→ Return best repaired code
```

## Tuning Guidelines

| Pass@1 Range | Recommended Action |
|--------------|-------------------|
| 0.9 - 1.0 | Basic repair (likely 1 small bug) |
| 0.7 - 0.9 | Basic repair with fallback |
| 0.3 - 0.7 | RL repair (multiple issues) |
| 0.0 - 0.3 | RL repair (fundamental problems) |

## Logs

The system will print which strategy is being used:
```
Using basic repair for problem 2999 (pass@1=0.90)
Basic repair succeeded, returning repaired code
```

Or:
```
Using basic repair for problem 2999 (pass@1=0.90)
Basic repair failed, falling back to RL repair
Generating synthetic tests for problem 2999...
```

## Benefits

1. **Faster Overall Pipeline**: High-pass-rate problems get quick fixes
2. **Resource Optimization**: Don't waste synthetic tests on simple bugs
3. **Better RL Training**: RL focuses on genuinely difficult problems
4. **Robust Fallback**: Always has a backup strategy
