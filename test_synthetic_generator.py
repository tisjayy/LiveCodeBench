#!/usr/bin/env python3
"""
Test script to debug synthetic test generation issues.
"""

import sys
import os
import json
import tempfile
import subprocess

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from lcb_runner.evaluation.synthetic_test_generator import SyntheticTestGenerator, TestCase
from lcb_runner.evaluation.internal_evaluator import InternalEvaluator
from lcb_runner.lm_styles import LanguageModel, LMStyle

# Mock language model for testing
class MockLanguageModel:
    def __init__(self):
        self.model_style = LMStyle.DeepSeekAPI
        self.model_repr = "test-model"

class MockBaseRunner:
    def prompts_to_outputs(self, prompts):
        # Mock response for test plan generation
        if "test plan" in prompts[0].lower():
            return [[json.dumps([
                {"category": "basic", "intent": "Simple case", "example_inputs": ["1\\n3 2\\nBWB"]},
                {"category": "boundary", "intent": "All black", "example_inputs": ["1\\n5 5\\nBBBBB"]},
                {"category": "edge", "intent": "No black cells", "example_inputs": ["1\\n3 2\\nWWW"]}
            ])]]
        
        # Mock response for concrete test generation
        elif "concrete" in prompts[0].lower():
            return [[json.dumps([
                {
                    "category": "basic",
                    "input": "1\\n3 2\\nBWB", 
                    "expected": "1",
                    "rationale": "One operation needed to cover the B at position 0"
                },
                {
                    "category": "boundary", 
                    "input": "1\\n5 5\\nBBBBB",
                    "expected": "1", 
                    "rationale": "One operation covers all black cells"
                },
                {
                    "category": "edge",
                    "input": "1\\n3 2\\nWWW",
                    "expected": "0",
                    "rationale": "No black cells to remove"
                }
            ])]]
        
        # Mock response for consistency checking
        elif "compute the expected output" in prompts[0].lower():
            if "1\\n3 2\\nBWB" in prompts[0]:
                return [["1"]]
            elif "1\\n5 5\\nBBBBB" in prompts[0]:
                return [["1"]]
            elif "1\\n3 2\\nWWW" in prompts[0]:
                return [["0"]]
        
        return [[""]]

def test_1873d_problem():
    """Test the 1D Eraser problem specifically."""
    
    question = """You are given a strip of paper s that is n cells long. Each cell is either black or white. In an operation you can take any k consecutive cells and make them all white.

Find the minimum number of operations needed to remove all black cells.

Input
The first line contains a single integer t (1 ≤ t ≤ 1000) — the number of test cases.
The first line of each test case contains two integers n and k (1 ≤ k ≤ n ≤ 2⋅10^5) — the length of the paper and the integer used in the operation.
The second line of each test case contains a string s of length n consisting of characters B (representing a black cell) or W (representing a white cell).

Output
For each test case, output a single integer — the minimum number of operations needed to remove all black cells."""

    # Initialize components
    model = MockLanguageModel()
    base_runner = MockBaseRunner()
    
    print("Testing Synthetic Test Generator...")
    test_generator = SyntheticTestGenerator(model, base_runner)
    
    # Generate tests
    synthetic_tests = test_generator.generate_tests(
        question=question,
        code_type="stdin",
        num_tests=3,
        self_consistency_checks=2,
        min_confidence=0.5
    )
    
    print(f"\nGenerated {len(synthetic_tests)} synthetic tests:")
    for i, test in enumerate(synthetic_tests):
        print(f"Test {i}:")
        print(f"  Category: {test.category}")
        print(f"  Input: {repr(test.input_val)}")
        print(f"  Expected: {repr(test.expected_output)}")
        print(f"  Confidence: {test.confidence}")
        print()
    
    # Test with a correct solution
    correct_solution = """
import sys

def main():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        s = input().strip()
        
        operations = 0
        i = 0
        while i < n:
            if s[i] == 'B':
                operations += 1
                i += k
            else:
                i += 1
        print(operations)

if __name__ == "__main__":
    main()
"""
    
    print("Testing with correct solution...")
    internal_evaluator = InternalEvaluator()
    
    results, metadata = internal_evaluator.evaluate_on_synthetic_tests(
        code=correct_solution,
        tests=synthetic_tests,
        code_type="stdin"
    )
    
    print(f"Results: {results}")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    # Test with an incorrect solution
    incorrect_solution = """
import sys

def main():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        s = input().strip()
        
        # Wrong approach - just count black cells
        black_count = s.count('B')
        print(black_count)

if __name__ == "__main__":
    main()
"""
    
    print("\nTesting with incorrect solution...")
    results2, metadata2 = internal_evaluator.evaluate_on_synthetic_tests(
        code=incorrect_solution,
        tests=synthetic_tests,
        code_type="stdin"
    )
    
    print(f"Results: {results2}")
    print(f"Passed: {sum(results2)}/{len(results2)}")
    
    return len(synthetic_tests) > 0 and sum(results) > sum(results2)

if __name__ == "__main__":
    success = test_1873d_problem()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")