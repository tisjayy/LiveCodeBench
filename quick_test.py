#!/usr/bin/env python3
"""
Simple test to verify the synthetic test generation fixes.
"""

# Test JSON parsing
test_json = '[{"category": "basic", "input": "1\\n3 2\\nBWB", "expected": "1", "rationale": "Basic test"}]'

try:
    import json
    parsed = json.loads(test_json)
    print(f"✅ JSON parsing works: {len(parsed)} tests parsed")
    print(f"Sample test input: {repr(parsed[0]['input'])}")
    print(f"Sample test expected: {repr(parsed[0]['expected'])}")
except Exception as e:
    print(f"❌ JSON parsing failed: {e}")

# Test output comparison
from lcb_runner.evaluation.internal_evaluator import InternalEvaluator

evaluator = InternalEvaluator()

# Test cases for comparison
test_cases = [
    ("1", "1", True),      # Exact match
    ("1\n", "1", True),    # With trailing newline
    ("1 ", "1", True),     # With trailing space
    ("2", "1", False),     # Different numbers
]

print("\n✅ Testing output comparison:")
for actual, expected, should_match in test_cases:
    result = evaluator._compare_outputs(actual, expected)
    status = "✅" if result == should_match else "❌"
    print(f"{status} '{actual}' vs '{expected}' -> {result} (expected {should_match})")

print("\n🔧 Key improvements made:")
print("1. Enhanced JSON parsing with fallback extraction")
print("2. Improved stdin input format handling") 
print("3. Better output comparison with normalization")
print("4. Added detailed logging for debugging")
print("\nThese fixes should resolve the synthetic test failures.")