"""
Unified Code Generation and Repair Runner.

This runner generates baseline and two repair variants in a single pass:
1. Generate Code_A for each problem
2. PATH 1: Evaluate Code_A on ground-truth (baseline)
3. PATH 2: Repair Code_A using RL strategy selection (rl_repair)
4. PATH 3: Repair Code_A using round-robin strategy selection (rr_repair)

No leakage: All repair decisions based only on synthetic tests, not ground-truth.
Fair comparison: RL and RR run independently with no shared information.
"""

import os
import json
import sys
import logging
import time
from tqdm import tqdm

from lcb_runner.lm_styles import LanguageModel
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.prompts.self_repair import format_prompt_self_repair, extract_code_with_llm_fallback
from lcb_runner.utils.extraction_utils import extract_code  # Use ground-truth extraction

# Import evaluation functions
from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations_by_problem

# Import RL components
from lcb_runner.runner.rl_agent import RLAgent

# Import synthetic test generation and internal evaluation
from lcb_runner.evaluation.synthetic_test_generator import SyntheticTestGenerator
from lcb_runner.evaluation.internal_evaluator import InternalEvaluator, compute_weighted_reward

# Define Action Space
PROMPT_ACTIONS = {
    0: "default",
    1: "step_by_step_reasoning",
    2: "error_specific_feedback",
    3: "syntax_check",
    4: "edge_case_check",
}
ACTION_DIM = len(PROMPT_ACTIONS)


class UnifiedCodeGenAndRepairRunner:
    """
    Unified runner that generates baseline and RL repair results in one pass.
    (RR repair currently disabled for testing RL improvements)
    Ensures fair comparison: same initial code.
    """
    
    def __init__(self, args, model: LanguageModel):
        self.args = args
        self.model = model  # Model for code generation and repair
        self._base_runner = None
        
        # Optional: Different model for test generation (stronger model recommended)
        self.test_gen_model = None
        self._test_gen_base_runner = None
        if hasattr(args, 'test_gen_model') and args.test_gen_model:
            from lcb_runner.lm_styles import LanguageModelStore
            try:
                self.test_gen_model = LanguageModelStore[args.test_gen_model]
                print(f"\n{'='*80}")
                print(f"✓ SUCCESS: Test generation model loaded")
                print(f"  Requested: {args.test_gen_model}")
                print(f"  Loaded: {self.test_gen_model.model_repr}")
                print(f"  Repair model: {model.model_repr}")
                print(f"  Using different models: {self.test_gen_model.model_repr != model.model_repr}")
                print(f"{'='*80}\n")
                logging.info(f"Test generation model: {self.test_gen_model.model_repr}, Repair model: {model.model_repr}")
            except KeyError:
                print(f"\n{'='*80}")
                print(f"✗ WARNING: Test generation model '{args.test_gen_model}' NOT FOUND in LanguageModelStore!")
                print(f"  Available GPT-4o models:")
                # Show available GPT-4o models
                from lcb_runner.lm_styles import LanguageModelStore
                gpt4o_models = [k for k in LanguageModelStore.keys() if 'gpt-4o' in k.lower()]
                for m in sorted(gpt4o_models):
                    print(f"    - {m}")
                print(f"  Falling back to using {model.model_repr} for both test generation and repair")
                print(f"{'='*80}\n")
                logging.warning(f"Test generation model '{args.test_gen_model}' not found, using {model.model_repr} for both")
                self.test_gen_model = model
        else:
            self.test_gen_model = model  # Use same model if not specified
            print(f"INFO: No --test_gen_model specified, using {model.model_repr} for both test generation and repair")
        
        # RL agent for self-repair path
        self.rl_agent = RLAgent(action_dim=ACTION_DIM)
        self.max_repair_attempts = getattr(args, 'max_repair_attempts', 10)  # Increased from 5 to give more chances for edge case fixes
        
        # Internal evaluator and test generator
        self.internal_evaluator = InternalEvaluator(timeout=args.timeout)
        self.test_generator = None
        
        # Cache for synthetic tests
        self.synthetic_test_cache = {}
    
    @property
    def base_runner(self):
        """Lazy initialization of base runner for repair."""
        if self._base_runner is None:
            from lcb_runner.runner.runner_utils import build_runner
            
            # Temporarily disable both unified and use_rl flags to get the base runner
            original_use_rl = self.args.use_rl
            original_unified = getattr(self.args, 'unified', False)
            self.args.use_rl = False
            self.args.unified = False
            try:
                self._base_runner = build_runner(self.args, self.model)
            finally:
                self.args.use_rl = original_use_rl
                self.args.unified = original_unified
        return self._base_runner
    
    @property
    def test_gen_base_runner(self):
        """Lazy initialization of base runner for test generation (uses stronger model if specified)."""
        if self._test_gen_base_runner is None:
            from lcb_runner.runner.runner_utils import build_runner
            
            # Log which model we're building the runner for
            print(f"  [Building test_gen_base_runner] Model: {self.test_gen_model.model_repr}", file=sys.stderr, flush=True)
            logging.info(f"Building test_gen_base_runner with model: {self.test_gen_model.model_repr}")
            
            # Temporarily disable both unified and use_rl flags to get the base runner
            original_use_rl = self.args.use_rl
            original_unified = getattr(self.args, 'unified', False)
            self.args.use_rl = False
            self.args.unified = False
            try:
                self._test_gen_base_runner = build_runner(self.args, self.test_gen_model)
                # Verify the runner was created with the correct model
                if hasattr(self._test_gen_base_runner, 'model'):
                    actual_model = self._test_gen_base_runner.model.model_repr
                    print(f"  [test_gen_base_runner created] Actual model in runner: {actual_model}", file=sys.stderr, flush=True)
                    if actual_model != self.test_gen_model.model_repr:
                        print(f"  ⚠️  WARNING: Model mismatch! Expected {self.test_gen_model.model_repr}, got {actual_model}", file=sys.stderr, flush=True)
            finally:
                self.args.use_rl = original_use_rl
                self.args.unified = original_unified
        return self._test_gen_base_runner
    
    def run_main(self, benchmark: list, format_prompt: callable):
        """
        Main execution: Generate baseline, RL repair, and RR repair results for all problems.
        Returns: (baseline_results, rl_results, rr_results)
        """
        # Initialize test generator (uses stronger model if specified)
        if self.test_generator is None:
            print(f"\n{'='*80}")
            print(f"TEST GENERATOR INITIALIZATION:")
            print(f"  Model for test generation: {self.test_gen_model.model_repr}")
            print(f"  Model for repair: {self.model.model_repr}")
            print(f"  Using different models: {self.test_gen_model.model_repr != self.model.model_repr}")
            print(f"{'='*80}\n")
            logging.info(f"Test generator using model: {self.test_gen_model.model_repr} (repair model: {self.model.model_repr})")
            
            self.test_generator = SyntheticTestGenerator(
                model=self.test_gen_model,  # Use test generation model (stronger if specified)
                base_runner=self.test_gen_base_runner,  # Use test generation runner
                timeout=self.args.timeout
            )
        
        # Load existing results if continue_existing flag is set
        baseline_results = []
        rl_results = []
        rr_results = []
        
        if getattr(self.args, 'continue_existing', False) or getattr(self.args, 'continue_existing_with_eval', False):
            output_dir = f"output/{self.model.model_repr}"
            temp_str = f"{self.args.temperature}"
            baseline_path = os.path.join(output_dir, f"baseline_{temp_str}.json")
            rl_path = os.path.join(output_dir, f"rl_repair_{temp_str}.json")
            rr_path = os.path.join(output_dir, f"rr_repair_{temp_str}.json")
            
            # Load existing results
            if os.path.exists(baseline_path):
                with open(baseline_path, 'r') as f:
                    baseline_results = json.load(f)
            if os.path.exists(rl_path):
                with open(rl_path, 'r') as f:
                    rl_results = json.load(f)
            if os.path.exists(rr_path):
                with open(rr_path, 'r') as f:
                    rr_results = json.load(f)
            
            # Filter benchmark to only include problems not yet processed
            if baseline_results:
                processed_ids = {r["question_id"] for r in baseline_results}
                original_len = len(benchmark)
                benchmark = [s for s in benchmark if s.question_id not in processed_ids]
                print(f"\n{'='*80}")
                print(f"Found {len(baseline_results)} existing generations, continuing with {len(benchmark)} remaining")
                print(f"{'='*80}\n")
                starting_count = len(baseline_results)
            else:
                baseline_results = []
                rl_results = []
                rr_results = []
                starting_count = 0
        else:
            starting_count = 0
        
        total_problems = starting_count + len(benchmark)
        
        for idx, sample in enumerate(tqdm(benchmark, desc="Baseline + RL Pipeline (RR disabled)"), start=starting_count+1):
            print(f"\n{'='*80}")
            print(f"Problem {sample.question_id}: {sample.question_title}")
            print(f"{'='*80}")
            
            # STEP 1: Generate initial code (same for all three paths)
            initial_code, initial_output = self.generate_initial_code(sample, format_prompt)
            
            if not initial_code or not initial_code.strip():
                print(f"  ✗ Failed to generate valid code")
                # All paths get empty result
                baseline_results.append(self.create_empty_result(sample, "baseline"))
                rl_results.append(self.create_empty_result(sample, "rl_repair"))
                rr_results.append(self.create_empty_result(sample, "rr_repair"))  # Placeholder
                continue
            
            print(f"  ✓ Generated initial code ({len(initial_code)} chars)")
            
            # PATH 1: Baseline (evaluate on ground-truth, no repair)
            baseline_result = self.run_baseline_path(sample, initial_code, initial_output)
            baseline_results.append(baseline_result)
            
            # PATH 2: RL Repair (synthetic tests → RL strategy selection → repair)
            print(f"\n[RL REPAIR] Starting repair process...")
            rl_result = self.run_selfrepair_path(
                sample, initial_code, initial_output, format_prompt, repair_type="RL"
            )
            rl_results.append(rl_result)
            
            # PATH 3: Round-Robin Repair (COMMENTED OUT - Testing RL only)
            # print(f"\n[RR REPAIR] Starting repair process...")
            # rr_result = self.run_selfrepair_path(
            #     sample, initial_code, initial_output, format_prompt, repair_type="RR"
            # )
            # rr_results.append(rr_result)
            rr_results.append(self.create_empty_result(sample, "rr_repair"))  # Placeholder
            
            # Save incremental results every 10 problems
            if idx % 10 == 0:
                print(f"\n{'='*80}")
                print(f"💾 Saving incremental results (completed {idx}/{total_problems} problems)...")
                print(f"{'='*80}")
                self.save_results(baseline_results, rl_results, rr_results)
        
        # Save final results
        self.save_results(baseline_results, rl_results, rr_results)
        
        # Save trained RL agent
        self.save_rl_agent()
        
        # Return for evaluation
        return [[r["output_list"][0]] for r in rl_results]
    
    def generate_initial_code(self, sample, format_prompt):
        """Generate initial code (same for both baseline and repair)."""
        prompt = format_prompt(sample, self.model.model_style)
        
        try:
            outputs = self.base_runner.prompts_to_outputs([prompt])
            output = outputs[0][0]
            code = extract_code(output, self.model.model_style)
            
            # Validate code completeness
            if code:
                print(f"  Generated code: {len(code)} chars, {len(output)} total output chars")
                print(f"\n  === FULL GENERATED CODE ===")
                # Print code with proper indentation for readability
                for line in code.split('\n'):
                    print(f"  {line}")
                print(f"  === END GENERATED CODE ===\n")
                logging.info(f"Generated initial code for {sample.question_id}: {len(code)} chars")
                logging.info(f"Full generated code:\n{code}")
                
                # Check for truncation indicators
                if self._is_code_truncated(code, sample):
                    print(f"  ⚠️  WARNING: Code appears truncated!")
                    print(f"  Last 100 chars: ...{code[-100:]}")
                    print(f"  Recommendation: Increase --max_tokens (current: {self.args.max_tokens})")
                    logging.warning(f"Code may be truncated for {sample.question_id}. "
                                  f"Consider increasing --max_tokens (current: {self.args.max_tokens})")
            
            return code, output
        except Exception as e:
            logging.error(f"Error generating code for {sample.question_id}: {e}")
            return None, None
    
    def run_baseline_path(self, sample, code, output):
        """
        PATH 1: Baseline evaluation.
        Evaluate initial code on ground-truth, no repair.
        """
        print(f"\n[BASELINE] Evaluating initial code on ground-truth...")
        
        # Evaluate on ground-truth
        eval_args = ([code], sample.get_evaluation_sample(), False, self.args.timeout)
        graded, metadata = evaluate_generations_by_problem(eval_args)
        
        if graded:
            passed_count = sum(graded[0])
            total_count = len(graded[0])
            passed = all(graded[0])
            print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'} ({passed_count}/{total_count} tests)")
        else:
            passed = False
            print(f"  Result: ✗ FAIL (evaluation error)")
        
        return {
            "question_id": sample.question_id,
            "question_title": sample.question_title,
            "code_list": [code],
            "output_list": [output],
            "graded_list": graded[0] if graded else [],
            "metadata": metadata,
            "path": "baseline",
            "pass@1": 1.0 if passed else 0.0
        }
    
    def run_selfrepair_path(self, sample, initial_code, initial_output, format_prompt, repair_type="RL"):
        """
        PATH 2/3: Repair path (RL or Round-Robin).
        Generate synthetic tests, repair if needed, evaluate on ground-truth.
        
        Args:
            repair_type: "RL" or "RR"
        """
        # Determine code type FIRST (before generating tests)
        code_type = self._determine_code_type(sample)
        
        # Map to evaluator format (internal_evaluator expects "stdin" or "call_based")
        eval_code_type = "call_based" if code_type == "function" else "stdin"
        
        # Generate ADVERSARIAL synthetic tests (targeted at initial code's weaknesses)
        # This follows UTRL's approach: tests designed to expose bugs in specific code
        synthetic_tests = self.generate_or_get_synthetic_tests(
            sample, code_type, adversarial_code=initial_code
        )
        
        if not synthetic_tests:
            print(f"  ✗ Failed to generate synthetic tests, using original code")
            baseline_result = self.run_baseline_path(sample, initial_code, initial_output)
            # Convert to repair result format
            repair_result = baseline_result.copy()
            repair_result["path"] = f"{repair_type.lower()}_repair"
            repair_result["repair_attempted"] = False
            repair_result["attempts_used"] = 0
            repair_result["synth_all_passed"] = False
            return repair_result
        
        # NEW: Check if test set is too weak (minimum requirements)
        if len(synthetic_tests) < 3:
            print(f"  ⚠️  Only {len(synthetic_tests)} tests generated (need ≥3 for reliable repair)")
            print(f"  → Skipping repair, using original code")
            baseline_result = self.run_baseline_path(sample, initial_code, initial_output)
            repair_result = baseline_result.copy()
            repair_result["path"] = f"{repair_type.lower()}_repair"
            repair_result["repair_attempted"] = False
            repair_result["attempts_used"] = 0
            repair_result["synth_all_passed"] = False
            repair_result["skip_reason"] = "insufficient_tests"
            return repair_result
        
        print(f"  ✓ Generated {len(synthetic_tests)} synthetic tests (format: {eval_code_type})")
        
        # NEW: Validate test quality - check category diversity
        test_categories = {}
        for test in synthetic_tests:
            cat = test.category
            test_categories[cat] = test_categories.get(cat, 0) + 1
        
        has_stress = test_categories.get('stress', 0) > 0
        has_edge = test_categories.get('edge', 0) > 0
        
        if not has_stress:
            print(f"  ⚠️  WARNING: No stress tests - repair may overfit to small inputs!")
            logging.warning("Synthetic test set missing stress tests - repair quality may be poor")
        
        if not has_edge:
            print(f"  ⚠️  WARNING: No edge case tests - repair may miss corner cases!")
            logging.warning("Synthetic test set missing edge case tests")
        
        # Evaluate initial code on SYNTHETIC tests
        synth_results, synth_metadata = self.internal_evaluator.evaluate_on_synthetic_tests(
            code=initial_code, tests=synthetic_tests, code_type=eval_code_type
        )
        
        passed_synth = sum(synth_results)
        total_synth = len(synth_results)
        print(f"  Initial code: {passed_synth}/{total_synth} synthetic tests passed")
        
        # NEW: Check if tests are TOO EASY (all pass baseline = useless for repair)
        if passed_synth == total_synth:
            print(f"  ⚠️  WARNING: ALL synthetic tests passed baseline!")
            print(f"     These tests won't drive repair (no failing tests to fix).")
            logging.warning("All synthetic tests passed baseline - tests may be too trivial for repair")
        
        # Decision: Repair or not (based on SYNTHETIC tests only)
        if all(synth_results):
            print(f"  ✓ Passed all synthetic tests, no repair needed")
            final_code = initial_code
            final_output = initial_output
            repair_attempted = False
            attempts_used = 0
            synth_all_passed = True
        else:
            # Calculate pass rate
            pass_rate = sum(synth_results) / len(synth_results) if synth_results else 0
            print(f"  → Running {repair_type} repair ({self.max_repair_attempts} max attempts)...")
            print(f"  Initial synthetic test pass rate: {sum(synth_results)}/{len(synth_results)} ({pass_rate:.1%})")
            
            # Run repair loop (RL or RR)
            final_code, final_output, attempts_used, synth_all_passed = self.run_rl_repair_loop(
                sample, initial_code, initial_output,
                synthetic_tests, synth_results, synth_metadata,
                eval_code_type, format_prompt, repair_type
            )
            repair_attempted = True
            print(f"  ✓ Repair complete (best solution found at attempt {attempts_used}/{self.max_repair_attempts})")
        
        # Evaluate final code on GROUND-TRUTH (first time seeing ground-truth!)
        print(f"  Evaluating final code on ground-truth...")
        eval_args = ([final_code], sample.get_evaluation_sample(), False, self.args.timeout)
        graded, metadata = evaluate_generations_by_problem(eval_args)
        
        if graded:
            passed_count = sum(graded[0])
            total_count = len(graded[0])
            passed = all(graded[0])
            print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'} ({passed_count}/{total_count} tests)")
        else:
            passed = False
            print(f"  Result: ✗ FAIL (evaluation error)")
        
        return {
            "question_id": sample.question_id,
            "question_title": sample.question_title,
            "code_list": [final_code],
            "output_list": [final_output],
            "graded_list": graded[0] if graded else [],
            "metadata": metadata,
            "path": f"{repair_type.lower()}_repair",
            "repair_attempted": repair_attempted,
            "attempts_used": attempts_used,
            "synth_all_passed": synth_all_passed,
            "initial_synth_passed": f"{passed_synth}/{total_synth}",
            "pass@1": 1.0 if passed else 0.0
        }
    
    def run_rl_repair_loop(self, sample, initial_code, initial_output,
                           synthetic_tests, synth_results, synth_metadata,
                           eval_code_type, format_prompt, repair_type="RL"):
        """
        Repair loop with RL or Round-Robin strategy selection.
        Early stopping when all synthetic tests pass.
        
        Args:
            eval_code_type: "stdin" or "call_based" (for evaluator)
            repair_type: "RL" or "RR"
            
        Returns: (final_code, final_output, attempts_used, synth_all_passed)
        """
        current_code = initial_code
        best_code = initial_code
        best_output = initial_output
        best_synth_passed = sum(synth_results)
        best_synth_results = synth_results.copy()
        best_synth_metadata = synth_metadata  # CRITICAL: Initialize metadata for tracking
        
        initial_test_confidences = [test.confidence for test in synthetic_tests]
        
        # Compute initial code's reward so we don't accept worse repairs
        initial_reward = compute_weighted_reward(
            test_results=synth_results,
            test_confidences=initial_test_confidences,
            old_results=None  # No previous results for initial code
        )
        best_reward = initial_reward
        best_attempt = 0  # Track when best solution was first found (0 = initial code)
        
        attempts_used = 0
        synth_all_passed = False
        
        for attempt in range(self.max_repair_attempts):
            attempts_used = attempt + 1
            
            # Get current failure metadata
            current_metadata = self._get_failure_metadata(
                synth_results if attempt == 0 else best_synth_results,
                synth_metadata if attempt == 0 else best_synth_metadata,
                synthetic_tests
            )
            
            # Select strategy (RL or Round-Robin)
            if repair_type == "RL":
                error_code = current_metadata.get("error_code", 0)
                
                state = {
                    "question": sample.question_content,
                    "code": current_code,
                    "metadata": json.dumps(current_metadata)
                }
                
                # 🎯 Enhanced cold start: Use error-specific hints for first 3 attempts
                # This ensures diversity and faster convergence
                if attempt < 3:
                    if error_code == -3:  # Timeout - use complexity-aware strategy
                        action_idx = 1  # step_by_step_reasoning (has timeout logic)
                        prompt_strategy = PROMPT_ACTIONS[action_idx]
                        print(f"    Attempt {attempt+1}: Strategy = {prompt_strategy} [RL+hint: timeout→complexity]")
                    elif error_code == -1:  # Syntax - use syntax checker
                        action_idx = 3  # syntax_check
                        prompt_strategy = PROMPT_ACTIONS[action_idx]
                        print(f"    Attempt {attempt+1}: Strategy = {prompt_strategy} [RL+hint: syntax→check]")
                    elif error_code == -4:  # Runtime - use error_specific
                        action_idx = 2  # error_specific_feedback
                        prompt_strategy = PROMPT_ACTIONS[action_idx]
                        print(f"    Attempt {attempt+1}: Strategy = {prompt_strategy} [RL+hint: runtime→specific]")
                    elif error_code == -2:  # Wrong answer - rotate through logical strategies
                        # Cycle through strategies that help with logic: default, step_by_step, error_specific
                        logic_strategies = [0, 1, 2]  # default, step_by_step, error_specific
                        action_idx = logic_strategies[attempt % len(logic_strategies)]
                        prompt_strategy = PROMPT_ACTIONS[action_idx]
                        print(f"    Attempt {attempt+1}: Strategy = {prompt_strategy} [RL+hint: wrong_answer→logic_{attempt}]")
                    else:
                        # For other cases (pass or unknown), use RL policy
                        action_idx = self.rl_agent.select_action(state)
                        prompt_strategy = PROMPT_ACTIONS[action_idx]
                        print(f"    Attempt {attempt+1}: Strategy = {prompt_strategy} [RL: exploring]")
                else:
                    # After 3 attempts, trust the learned policy
                    action_idx = self.rl_agent.select_action(state)
                    prompt_strategy = PROMPT_ACTIONS[action_idx]
                    print(f"    Attempt {attempt+1}: Strategy = {prompt_strategy} [RL: learned policy]")
            else:  # Round-Robin
                action_idx = attempt % ACTION_DIM
                prompt_strategy = PROMPT_ACTIONS[action_idx]
                print(f"    Attempt {attempt+1}: Strategy = {prompt_strategy} [RR]")
            
            # Generate repair with selected strategy
            prompt = format_prompt_self_repair(
                question=sample.question_content,
                LanguageModelStyle=self.model.model_style,
                code=current_code,
                result=False,
                metadata=json.dumps(current_metadata),
                strategy=prompt_strategy
            )
            
            outputs = self.base_runner.prompts_to_outputs([prompt])
            model_output = outputs[0][0]
            repaired_code = extract_code_with_llm_fallback(model_output, self.model.model_style, self.base_runner)
            
            # Log repaired code
            if repaired_code:
                print(f"      Repaired code: {len(repaired_code)} chars")
                print(f"\n      === FULL REPAIRED CODE (Attempt {attempt+1}) ===")
                # Print code with proper indentation for readability
                for line in repaired_code.split('\n'):
                    print(f"      {line}")
                print(f"      === END REPAIRED CODE ===\n")
                logging.info(f"Attempt {attempt+1} repaired code for {sample.question_id}: {len(repaired_code)} chars")
                logging.info(f"Full repaired code (attempt {attempt+1}):\n{repaired_code}")
            else:
                print(f"      ⚠️  WARNING: Failed to extract code from repair response")
                logging.warning(f"Attempt {attempt+1} failed to extract code for {sample.question_id}")
            
            # Evaluate repaired code on synthetic tests
            new_synth_results, new_synth_metadata = self.internal_evaluator.evaluate_on_synthetic_tests(
                code=repaired_code, tests=synthetic_tests, code_type=eval_code_type
            )
            
            passed_synth = sum(new_synth_results)
            
            # Compute reward for RL (for tracking improvement)
            if repair_type == "RL":
                reward = compute_weighted_reward(
                    test_results=new_synth_results,
                    test_confidences=initial_test_confidences,
                    old_results=best_synth_results
                )
                self.rl_agent.store_reward(reward)
            else:
                # RR doesn't use rewards, just track for best selection
                reward = compute_weighted_reward(
                    test_results=new_synth_results,
                    test_confidences=initial_test_confidences,
                    old_results=best_synth_results
                )
            
            # Update best if improved (more tests passed, or same tests but better reward)
            if passed_synth > best_synth_passed or (passed_synth == best_synth_passed and reward > best_reward):
                best_code = repaired_code
                best_output = model_output
                best_synth_passed = passed_synth
                best_reward = reward
                best_synth_results = new_synth_results.copy()
                best_synth_metadata = new_synth_metadata
                best_attempt = attempt + 1  # Record when best was found
                if repair_type == "RL":
                    print(f"      ★ NEW BEST: {passed_synth}/{len(synth_results)} tests (reward={reward:.3f})")
                else:
                    print(f"      ★ NEW BEST: {passed_synth}/{len(synth_results)} tests")
            else:
                if repair_type == "RL":
                    print(f"      {passed_synth}/{len(synth_results)} tests (reward={reward:.3f})")
                else:
                    print(f"      {passed_synth}/{len(synth_results)} tests")
            
            # Early stopping: All synthetic tests pass
            if all(new_synth_results):
                print(f"      ✓ All {len(synthetic_tests)} synthetic tests passed!")
                synth_all_passed = True
                break
            
            # Use best code for next iteration
            current_code = best_code
        
        # Update RL policy after episode (RL only)
        if repair_type == "RL":
            try:
                self.rl_agent.update_policy()
            except Exception as e:
                logging.warning(f"Policy update failed: {e}")
        
        # Return best_attempt (when best was found) not attempts_used (total attempts run)
        # This gives fairer efficiency comparison: "attempts to find solution" vs "total attempts wasted"
        return best_code, best_output, best_attempt, synth_all_passed
    
    def generate_or_get_synthetic_tests(self, sample, code_type, adversarial_code=None):
        """Generate or retrieve cached synthetic tests with adaptive retry logic.
        
        Args:
            adversarial_code: If provided, generates tests targeted at exposing bugs in this code
        """
        # Don't cache adversarial tests (they're code-specific)
        if not adversarial_code and sample.question_id in self.synthetic_test_cache:
            return self.synthetic_test_cache[sample.question_id]
        
        # Map internal code_type to test generator format
        test_gen_type = "call_based" if code_type == "function" else "stdin"
        
        # Adaptive strategy: Reduce complexity on timeout
        # Attempt 1: 5 tests, 3 consistency checks (default/optimal)
        # Attempt 2: 3 tests, 1 consistency check (if timeout)
        # Attempt 3: 2 tests, 0 consistency checks (if still timing out)
        # Attempt 4: 1 test, 0 consistency checks (minimal fallback)
        test_configs = [
            {"num_tests": 5, "checks": 3, "delay": 5},   # Attempt 1 (default)
            {"num_tests": 3, "checks": 1, "delay": 10},  # Attempt 2 (reduced)
            {"num_tests": 2, "checks": 0, "delay": 10},  # Attempt 3 (further reduced)
            {"num_tests": 1, "checks": 0, "delay": 15},  # Attempt 4 (minimal)
        ]
        
        mode_str = "adversarial" if adversarial_code else "standard"
        
        for attempt, config in enumerate(test_configs):
            try:
                print(f"  Generating {config['num_tests']} {mode_str} synthetic tests (attempt {attempt + 1}/{len(test_configs)})...")
                tests = self.test_generator.generate_tests(
                    question=sample.question_content,
                    code_type=test_gen_type,
                    num_tests=config['num_tests'],
                    self_consistency_checks=config['checks'],
                    min_confidence=0.5,
                    adversarial_code=adversarial_code  # NEW: Enable adversarial mode
                )
                
                if tests:
                    # Only cache non-adversarial tests (adversarial tests are code-specific)
                    if not adversarial_code:
                        self.synthetic_test_cache[sample.question_id] = tests
                    print(f"  ✓ Successfully generated {len(tests)} tests")
                    return tests
                
            except Exception as e:
                if attempt < len(test_configs) - 1:
                    logging.warning(f"Test generation attempt {attempt+1} failed: {e}")
                    print(f"  ⚠️  Timeout with {config['num_tests']} tests. Retrying with fewer tests in {config['delay']}s...")
                    import time
                    time.sleep(config['delay'])
                else:
                    logging.error(f"Failed to generate synthetic tests after {len(test_configs)} attempts with reduced complexity: {e}")
                    print(f"  ✗ Could not generate synthetic tests even with minimal settings")
                    return None
        
        return None
    
    def _determine_code_type(self, sample):
        """Determine if problem is function-based or script-based."""
        starter = getattr(sample, 'starter_code', '')
        code_type = 'function' if starter and starter.strip() else 'script'
        
        # Debug logging
        print(f"  Code type: {code_type} (starter_code present: {bool(starter and starter.strip())})")
        
        return code_type
    
    def _is_code_truncated(self, code, sample):
        """Check if generated code appears to be truncated."""
        code = code.strip()
        
        # Check 1: Code ends abruptly (no proper closing)
        if code and not code.endswith(('pass', 'return', '}', ')', '\n')):
            # Check if last line is incomplete
            last_line = code.split('\n')[-1].strip()
            if last_line and not last_line.endswith((':',  ',', ')', '}', 'pass', 'return')):
                return True
        
        # Check 2: For class-based code, check if class definition is incomplete
        if 'class Solution' in code or 'class ' in code:
            # Count opening and closing braces/parens (Python doesn't use braces, but check anyway)
            if code.count('def ') > 0:
                # Check if there's at least one complete method
                lines = code.split('\n')
                in_def = False
                has_complete_def = False
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('def '):
                        in_def = True
                    elif in_def and (stripped.startswith('return') or stripped == 'pass'):
                        has_complete_def = True
                        in_def = False  # Reset for next method
                
                # Only warn if NO methods are complete
                if not has_complete_def:
                    return True
        
        # Check 3: Code length suspiciously close to typical token limits
        # Assuming ~4 chars per token, 5000 tokens ≈ 20000 chars
        if len(code) > (self.args.max_tokens * 3.5):  # Close to limit
            logging.warning(f"Code length {len(code)} chars is close to max_tokens limit")
            return True
        
        return False
    
    def _get_failure_metadata(self, synth_results, synth_metadata, synthetic_tests):
        """Extract comprehensive metadata for failing tests."""
        try:
            # Find first failing test
            fail_index = synth_results.index(False)
            failing_test = synthetic_tests[fail_index]
            failing_test_result = synth_metadata["test_results"][fail_index]
            
            # Extract the actual test metadata (nested inside)
            failing_metadata = failing_test_result.get("metadata", {})
            
            # Get error_code from metadata (set by evaluator)
            error_code = failing_metadata.get("error_code", -2)  # Default to wrong answer
            error_msg = failing_metadata.get("error", "")
            has_error = bool(error_msg)
            
            # Build comprehensive metadata
            metadata = {
                "inputs": str(failing_test.input_val),
                "output": str(failing_metadata.get("actual", "N/A")),
                "expected": str(failing_test.expected_output),
                "error_code": error_code,
                "test_category": failing_test.category,
                "test_confidence": failing_test.confidence,
            }
            
            # Add input size hint for timeout/complexity issues
            if error_code == -3:  # Timeout
                try:
                    inp_str = str(failing_test.input_val)
                    if '\n' in inp_str:
                        first_line = inp_str.split('\n')[0].strip()
                        if first_line.isdigit():
                            input_size = int(first_line)
                            metadata["input_size"] = input_size
                            # Add complexity hint
                            if input_size > 100000:
                                metadata["complexity_hint"] = "Need O(n) or O(n log n) - avoid O(n²)"
                            elif input_size > 10000:
                                metadata["complexity_hint"] = "Avoid O(n²) - use O(n log n) or better"
                except:
                    pass
            
            # Add error message if exists
            if has_error:
                # Truncate very long errors
                metadata["error"] = error_msg[:500] if len(error_msg) > 500 else error_msg
            
            # Add info about ALL failing tests (up to 3 examples for context)
            failed_indices = [i for i, passed in enumerate(synth_results) if not passed]
            if len(failed_indices) > 0:
                metadata["total_failed"] = len(failed_indices)
                metadata["failed_categories"] = list(set(synthetic_tests[i].category for i in failed_indices))
                
                # Add up to 3 failing test examples for better context
                if len(failed_indices) > 1:
                    other_failures = []
                    for idx in failed_indices[1:4]:  # Show up to 3 more examples (we already showed first)
                        test_result = synth_metadata["test_results"][idx]
                        test_meta = test_result.get("metadata", {})
                        other_failures.append({
                            "input": str(synthetic_tests[idx].input_val),
                            "expected": str(synthetic_tests[idx].expected_output),
                            "actual": str(test_meta.get("actual", "N/A")),
                            "category": synthetic_tests[idx].category
                        })
                    if other_failures:
                        metadata["other_failures"] = other_failures
            
            return metadata
            
        except (ValueError, IndexError, KeyError) as e:
            # All tests passed or error accessing metadata
            return {}
    
    def create_empty_result(self, sample, path="baseline"):
        """Create empty result for failed code generation."""
        return {
            "question_id": sample.question_id,
            "question_title": sample.question_title,
            "code_list": [""],
            "output_list": [""],
            "graded_list": [],
            "metadata": {},
            "path": path,
            "pass@1": 0.0
        }
    
    def save_results(self, baseline_results, rl_results, rr_results):
        """Save baseline, RL, and RR results."""
        output_dir = f"output/{self.model.model_repr}"
        os.makedirs(output_dir, exist_ok=True)
        
        temp_str = f"{self.args.temperature}"
        baseline_path = os.path.join(output_dir, f"baseline_{temp_str}.json")
        rl_path = os.path.join(output_dir, f"rl_repair_{temp_str}.json")
        rr_path = os.path.join(output_dir, f"rr_repair_{temp_str}.json")
        
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        
        with open(rl_path, 'w') as f:
            json.dump(rl_results, f, indent=2)
        
        with open(rr_path, 'w') as f:
            json.dump(rr_results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ Saved baseline results to: {baseline_path}")
        print(f"✓ Saved RL repair results to: {rl_path}")
        print(f"✓ Saved RR repair results to: {rr_path} (placeholder - RR disabled)")
        print(f"{'='*80}")
    
    def save_rl_agent(self):
        """Save trained RL agent."""
        save_dir = f"output/{self.model.model_repr}"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"rl_agent_unified_{self.max_repair_attempts}attempts.pt")
        
        self.rl_agent.save(save_path)
        print(f"\n✓ Saved trained RL agent to: {save_path}")
        print(f"  Strategy usage: {self.rl_agent.get_strategy_usage()}\n")
