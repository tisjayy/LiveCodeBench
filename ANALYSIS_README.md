# Self-Repair Analysis Script

This script analyzes the results of RL-based self-repair runs and compares them with the original code generation results.

## Usage

```bash
python analyze_selfrepair_results.py --model MODEL_NAME --codegen_n NUM --temperature TEMP
```

### Parameters

- `--model`: Name of the model (e.g., `DeepSeekCoder-V2.5`, `GPT-4O-mini-2024-07-18`)
- `--codegen_n`: Number of codegen samples (default: 10)
- `--temperature`: Temperature used for generation (default: 0.2)

### Examples

**Analyze DeepSeek-Coder results:**
```bash
python analyze_selfrepair_results.py --model DeepSeekCoder-V2.5 --codegen_n 10 --temperature 0.2
```

**Analyze GPT-4O-mini results:**
```bash
python analyze_selfrepair_results.py --model GPT-4O-mini-2024-07-18 --codegen_n 10 --temperature 0.2
```

## What it does

1. **Loads results**: Reads both the original codegen results and the selfrepair results
2. **Evaluates selfrepair**: Runs the test cases on the selfrepaired code
3. **Compares performance**: Shows before/after pass rates
4. **Analyzes strategies**: Detects which RL repair strategies were used:
   - `step_by_step_reasoning`: Uses step-by-step analysis
   - `error_specific_feedback`: Focuses on specific failing test cases
   - `edge_case_check`: Addresses edge cases and boundary conditions
   - `syntax_check`: Fixes syntax/compilation errors
   - `default`: Major code restructuring
   - `unknown`: Minor changes without clear indicators
5. **Reports regressions**: Identifies any problems that were fixed but later broke

## Output

The script provides:
- Original vs selfrepair pass rates
- Number of problems fixed/broken
- Strategy distribution for successful repairs
- List of fixed problems with their strategies
- List of broken problems (if any)

## Example Output

```
================================================================================
DEEPSEEKCODER-V2.5 RESULTS
================================================================================

ORIGINAL CODEGEN:
  Passed: 94/100 (94%)
  Failed: 6/100 (6%)

AFTER SELFREPAIR:
  Passed: 98/100 (98%)
  Failed: 2/100 (2%)

IMPROVEMENT: +4 problems (+4 fixed)

Pass@1 rate: 0.9800
================================================================================

STRATEGY ANALYSIS:
--------------------------------------------------------------------------------

PROBLEMS THAT WERE FIXED:
--------------------------------------------------------------------------------
1. 1873_D - D. 1D Eraser
   Strategy: step_by_step_reasoning
2. 1883_C - C. Raspberries
   Strategy: error_specific_feedback
...

Strategy Distribution (4 fixes):
  step_by_step_reasoning: 3 (75.0%)
  error_specific_feedback: 1 (25.0%)
```
