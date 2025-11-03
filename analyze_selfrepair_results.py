#!/usr/bin/env python3
"""
Analyze selfrepair results and compare with original codegen results.

Usage:
    python analyze_selfrepair_results.py --model DeepSeekCoder-V2.5 --codegen_n 10 --temperature 0.2
"""

import json
import sys
import argparse
import os
import difflib
sys.path.insert(0, '.')

from lcb_runner.benchmarks import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics


def load_results(model_name, codegen_n, temperature, output_dir_override=None):
    """Load codegen and selfrepair results."""
    if output_dir_override:
        output_dir = output_dir_override
    else:
        output_dir = f"output/{model_name}"
    
    codegen_file = f"{output_dir}/Scenario.codegeneration_{codegen_n}_{temperature}_eval_all.json"
    selfrepair_file = f"{output_dir}/Scenario.selfrepair_{codegen_n}_{temperature}.json"
    
    if not os.path.exists(codegen_file):
        raise FileNotFoundError(f"Codegen file not found: {codegen_file}")
    
    if not os.path.exists(selfrepair_file):
        raise FileNotFoundError(f"Selfrepair file not found: {selfrepair_file}")
    
    with open(codegen_file, 'r') as f:
        codegen_results = json.load(f)
    
    with open(selfrepair_file, 'r') as f:
        selfrepair_results = json.load(f)
    
    return codegen_results, selfrepair_results


def detect_strategy(original_code, repaired_code, output):
    """Try to detect which strategy was used based on the output."""
    output_lower = output.lower()
    
    # Check for step-by-step reasoning
    if any(keyword in output_lower for keyword in ['step', 'reason', 'analysis', 'first', 'then', 'finally']):
        return "step_by_step_reasoning"
    
    # Check for syntax error handling
    if any(keyword in output_lower for keyword in ['syntax', 'indentation', 'missing', 'typo', 'compile']):
        return "syntax_check"
    
    # Check for edge case mentions
    if any(keyword in output_lower for keyword in ['edge case', 'corner case', 'boundary', 'empty', 'large']):
        return "edge_case_check"
    
    # Check for specific error feedback
    if any(keyword in output_lower for keyword in ['error', 'failed', 'test case', 'input', 'output', 'expected']):
        return "error_specific_feedback"
    
    # Check if code had major changes
    code_diff_ratio = len(set(repaired_code) - set(original_code)) / max(len(original_code), 1)
    if code_diff_ratio > 0.5:  # Major restructuring
        return "default"
    
    return "unknown"


def analyze_results(codegen_results, selfrepair_results, model_name):
    """Analyze selfrepair results and compare with original."""
    # Load benchmark dataset
    dataset = load_code_generation_dataset()
    dataset = sorted(dataset, key=lambda x: x.question_id)
    
    # Sort results
    codegen_results = sorted(codegen_results, key=lambda x: x['question_id'])
    selfrepair_results = sorted(selfrepair_results, key=lambda x: x['question_id'])
    
    print(f"\nLoaded {len(dataset)} problems from benchmark")
    print(f"Loaded {len(codegen_results)} codegen results")
    print(f"Loaded {len(selfrepair_results)} selfrepair results\n")
    
    # Extract code lists from selfrepair
    selfrepair_code_lists = []
    for result in selfrepair_results:
        selfrepair_code_lists.append(result['code_list'])
    
    # Evaluate selfrepair results
    print("Evaluating selfrepair results...")
    metrics = codegen_metrics(
        [d.get_evaluation_sample() for d in dataset],
        selfrepair_code_lists,
        num_process_evaluate=8,
        timeout=6,
        debug=False
    )
    
    # Get graded results
    from lcb_runner.evaluation import extract_instance_results
    graded = extract_instance_results(metrics[1])
    
    print("\n" + "="*80)
    print(f"{model_name.upper()} RESULTS")
    print("="*80)
    
    # Count passes and failures
    selfrepair_passed = sum(1 for g in graded if all(g))
    selfrepair_failed = sum(1 for g in graded if not all(g))
    
    # Compare with original
    original_passed = sum(1 for c in codegen_results if c['graded_list'][0])
    original_failed = len(codegen_results) - original_passed
    
    print(f"\nORIGINAL CODEGEN:")
    print(f"  Passed: {original_passed}/100 ({original_passed}%)")
    print(f"  Failed: {original_failed}/100 ({original_failed}%)")
    
    print(f"\nAFTER SELFREPAIR:")
    print(f"  Passed: {selfrepair_passed}/100 ({selfrepair_passed}%)")
    print(f"  Failed: {selfrepair_failed}/100 ({selfrepair_failed}%)")
    
    improvement = selfrepair_passed - original_passed
    print(f"\nIMPROVEMENT: {improvement:+d} problems ({'+' if improvement > 0 else ''}{improvement} fixed)")
    
    print(f"\nPass@1 rate: {metrics[0]['pass@1']:.4f}")
    print("="*80)
    
    # Analyze strategies
    print("\nSTRATEGY ANALYSIS:")
    print("-"*80)
    
    # Strategy counters
    strategies_detected = {
        "step_by_step_reasoning": 0,
        "error_specific_feedback": 0,
        "syntax_check": 0,
        "edge_case_check": 0,
        "default": 0,
        "unknown": 0
    }
    
    # Problems fixed
    print("\nPROBLEMS THAT WERE FIXED:")
    print("-"*80)
    fixed_count = 0
    for i, (codeg, selfr, graded_list) in enumerate(zip(codegen_results, selfrepair_results, graded)):
        original_pass = codeg['graded_list'][0]
        selfrepair_pass = all(graded_list)
        
        if not original_pass and selfrepair_pass:
            fixed_count += 1
            problem_id = codeg['question_id']
            print(f"\n[{fixed_count}] Problem {problem_id} - FIXED")
            
            original_code = codeg['code_list'][0]
            repaired_code = selfr['code_list'][0]
            
            diff = difflib.unified_diff(
                original_code.splitlines(keepends=True),
                repaired_code.splitlines(keepends=True),
                fromfile='original',
                tofile='repaired',
            )
            
            print("Code difference:")
            sys.stdout.writelines(diff)
            
            # Detect strategy
            strategy = detect_strategy(original_code, repaired_code, selfr.get('output', ''))
            strategies_detected[strategy] += 1
            print(f"Detected strategy: {strategy}")
            print("-"*80)
            fixed_count += 1
            strategy = detect_strategy(
                codeg['code_list'][0],
                selfr['code_list'][0],
                selfr['output_list'][0]
            )
            strategies_detected[strategy] += 1
            print(f"{fixed_count}. {codeg['question_id']} - {codeg['question_title']}")
            print(f"   Strategy: {strategy}")
    
    if fixed_count == 0:
        print("(No problems were fixed)")
    
    print(f"\nStrategy Distribution ({sum(strategies_detected.values())} fixes):")
    for strategy, count in strategies_detected.items():
        if count > 0:
            pct = count / sum(strategies_detected.values()) * 100
            print(f"  {strategy}: {count} ({pct:.1f}%)")
    
    # Problems that broke
    print("\nPROBLEMS THAT BROKE:")
    print("-"*80)
    broken_count = 0
    for i, (codeg, selfr, graded_list) in enumerate(zip(codegen_results, selfrepair_results, graded)):
        original_pass = codeg['graded_list'][0]
        selfrepair_pass = all(graded_list)
        
        if original_pass and not selfrepair_pass:
            broken_count += 1
            print(f"{broken_count}. {codeg['question_id']} - {codeg['question_title']}")
    
    if broken_count == 0:
        print("(No problems broke)")
    
    print("\n" + "="*80)
    
    return {
        'original_passed': original_passed,
        'selfrepair_passed': selfrepair_passed,
        'improvement': improvement,
        'strategies': strategies_detected
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze selfrepair results")
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., DeepSeekCoder-V2.5)')
    parser.add_argument('--codegen_n', type=int, default=10, help='Number of codegen samples')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature')
    parser.add_argument('--output_dir', type=str, default=None, help='Override the output directory path')
    
    args = parser.parse_args()
    
    try:
        codegen_results, selfrepair_results = load_results(args.model, args.codegen_n, args.temperature, args.output_dir)
        analyze_results(codegen_results, selfrepair_results, args.model)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
