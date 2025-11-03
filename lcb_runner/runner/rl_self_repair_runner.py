import os
import json
import logging
from tqdm import tqdm

from lcb_runner.lm_styles import LanguageModel
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.prompts.self_repair import format_prompt_self_repair, extract_code

# --- 1. Import the correct, existing evaluation function ---
from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations_by_problem

# --- 2. Import our RL components ---
from lcb_runner.runner.rl_agent import RLAgent

# --- 3. Import synthetic test generation and internal evaluation ---
from lcb_runner.evaluation.synthetic_test_generator import SyntheticTestGenerator, TestCase
from lcb_runner.evaluation.internal_evaluator import InternalEvaluator, compute_weighted_reward

# --- Define Action Space and Map Actions to Strategies ---
PROMPT_ACTIONS = {
    0: "default",
    1: "step_by_step_reasoning",
    2: "error_specific_feedback",
    3: "syntax_check",
    4: "edge_case_check",
}
ACTION_DIM = len(PROMPT_ACTIONS)

class RLSelfRepairRunner:
    """
    A dedicated runner for the self-repair scenario that uses Reinforcement Learning.
    This runner is called by `build_runner` when --scenario is selfrepair and --use_rl is active.
    """
    def __init__(self, args, model: LanguageModel):
        self.args = args
        self.model = model
        self._base_runner = None  # Lazy initialization
        
        self.rl_agent = RLAgent(action_dim=ACTION_DIM)
        self.max_repair_attempts = getattr(args, 'max_repair_attempts', 5)
        
        # Initialize internal evaluator (test generator will be initialized after base_runner)
        self.internal_evaluator = InternalEvaluator(timeout=args.timeout)
        self.test_generator = None  # Will be initialized after base_runner
        
        # Cache for synthetic tests per question_id
        self.synthetic_test_cache = {}

        # Construct the path to the base code generation results file
        codegen_eval_path = f"output/{self.model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
        if not os.path.exists(codegen_eval_path):
            raise FileNotFoundError(f"Base codegen results not found at {codegen_eval_path}. Please run a codegen scenario first.")
        
        logging.info(f"Loading base codegen results from: {codegen_eval_path}")
        with open(codegen_eval_path, 'r') as f:
            self.codegen_results_map = {res['question_id']: res for res in json.load(f)}

    @property
    def base_runner(self):
        """Lazy initialization of the base runner to avoid circular dependencies."""
        if self._base_runner is None:
            from lcb_runner.runner.runner_utils import build_runner
            
            # Build a temporary base runner (not the RL one) for making model calls
            # Temporarily disable use_rl to get the normal runner
            original_use_rl = self.args.use_rl
            self.args.use_rl = False
            try:
                self._base_runner = build_runner(self.args, self.model)
            finally:
                self.args.use_rl = original_use_rl
        return self._base_runner

    def run_main(self, benchmark: list, format_prompt: callable) -> list[list[str]]:
        """
        Main execution method called by `eval_benchmark.py`.
        It runs the RL loop for each problem and returns the final raw model outputs.
        """
        all_final_outputs = []
        for sample in tqdm(benchmark, desc="RL Self-Repair Runner"):
            if sample.question_id not in self.codegen_results_map:
                logging.warning(f"Could not find base result for {sample.question_id}. Skipping.")
                all_final_outputs.append([""]) # Append empty string if no base code
                continue
            
            base_result = self.codegen_results_map[sample.question_id]
            final_model_output = self.run_sample_episode(sample, base_result, format_prompt)
            all_final_outputs.append([final_model_output])
            
        return all_final_outputs

    def run_sample_episode(self, sample, base_result, format_prompt) -> str:
        """
        Runs a full RL episode for a single problem using synthetic tests for reward.
        Ensures that the RL agent's policy is updated at the end of each episode.
        """
        # We only try to repair the first generated code sample (as n=1 is supported for self-repair)
        code = base_result["code_list"][0]
        passed = base_result["graded_list"][0]
        original_model_output = base_result["output_list"][0]
        final_model_output = original_model_output

        if passed:
            return original_model_output

        try:
            # Always clear memory at the start of an episode
            self.rl_agent.clear_memory()
            
            # Step 1: Generate or retrieve cached synthetic tests
            if sample.question_id in self.synthetic_test_cache:
                print(f"Using cached synthetic tests for problem {sample.question_id}")
                synthetic_tests = self.synthetic_test_cache[sample.question_id]
            else:
                print(f"Generating synthetic tests for problem {sample.question_id}...")
                if self.test_generator is None:
                    self.test_generator = SyntheticTestGenerator(self.model, self.base_runner)
                
                synthetic_tests = self.test_generator.generate_tests(
                    question=sample.question_content,
                    code_type=self._determine_code_type(sample),
                    num_tests=getattr(self.args, 'num_synthetic_tests', 5),
                    self_consistency_checks=getattr(self.args, 'self_consistency_checks', 3),
                    min_confidence=getattr(self.args, 'min_test_confidence', 0.5)
                )
                self.synthetic_test_cache[sample.question_id] = synthetic_tests
            
            if not synthetic_tests:
                logging.warning(f"Failed to generate synthetic tests for {sample.question_id}")
                return original_model_output
            
            code_type = self._determine_code_type(sample)
            current_code = code
            
            # Step 2: Evaluate initial code on synthetic tests
            initial_synth_results, initial_synth_metadata = self.internal_evaluator.evaluate_on_synthetic_tests(
                code=current_code, tests=synthetic_tests, code_type=code_type
            )
            new_synth_results = initial_synth_results

            # --- Start of RL Repair Loop ---
            # Always run at least one repair attempt for consistency, even if initial tests pass
            # This ensures we always have a final_model_output set correctly
            current_metadata = {}
            
            if all(initial_synth_results):
                print(f"Initial code for {sample.question_id} passed all {len(synthetic_tests)} synthetic tests. Running minimal RL pass for consistency.")
                # Still run one iteration with a dummy action for consistency
                # but this should immediately succeed and break
            else:
                try:
                    fail_index = initial_synth_results.index(False)
                    failing_test = synthetic_tests[fail_index]
                    failing_metadata = initial_synth_metadata["test_results"][fail_index]
                    current_metadata = {
                        "inputs": failing_test.input_val, "actual": failing_metadata.get("actual", ""),
                        "expected": failing_test.expected_output, "error_code": -2
                    }
                except (ValueError, IndexError, KeyError) as e:
                    logging.warning(f"Could not find failing synthetic test metadata: {e}")
                    current_metadata = {}
            
            initial_test_confidences = [test.confidence for test in synthetic_tests]
            current_synth_metadata = initial_synth_metadata  # Track latest metadata for diagnostics
            
            # Track the best code across all attempts
            best_code = current_code
            best_model_output = original_model_output
            best_synth_passed = sum(initial_synth_results)
            best_reward = -1.0  # Initialize with worst possible reward
            
            attempt_pbar = tqdm(range(self.max_repair_attempts), desc="Repair attempts", leave=False)
            for attempt in attempt_pbar:
                state = {
                    "question": sample.question_content, "code": current_code, "metadata": current_metadata,
                    "synthetic_test_summary": {
                        "total": len(synthetic_tests), "categories": list(set(t.category for t in synthetic_tests)),
                        "internal_diagnostics": self._extract_internal_diagnostics(current_synth_metadata) # Pass latest metadata
                    }
                }
                action_idx = self.rl_agent.select_action(state)
                prompt_strategy = PROMPT_ACTIONS[action_idx]
                attempt_pbar.set_description(f"Attempt {attempt+1}/{self.max_repair_attempts}: {prompt_strategy}")
                
                prompt = format_prompt(
                    question=sample.question_content, LanguageModelStyle=self.model.model_style,
                    code=current_code, result=False, metadata=json.dumps(current_metadata), strategy=prompt_strategy
                )
                
                outputs = self.base_runner.prompts_to_outputs([prompt])
                model_output = outputs[0][0]
                final_model_output = model_output
                repaired_code = extract_code(model_output, self.model.model_style)
                
                new_synth_results, new_synth_metadata = self.internal_evaluator.evaluate_on_synthetic_tests(
                    code=repaired_code, tests=synthetic_tests, code_type=code_type
                )
                
                reward = compute_weighted_reward(
                    test_results=new_synth_results, test_confidences=initial_test_confidences, old_results=initial_synth_results
                )
                self.rl_agent.store_reward(reward)
                
                passed_synth = sum(new_synth_results)
                total_synth = len(new_synth_results)
                
                # Update best code if this attempt is better
                # Priority: 1) More tests passed, 2) Higher reward if same number passed
                if passed_synth > best_synth_passed or (passed_synth == best_synth_passed and reward > best_reward):
                    best_code = repaired_code
                    best_model_output = model_output
                    best_synth_passed = passed_synth
                    best_reward = reward
                    print(f"  ★ NEW BEST: Attempt {attempt+1} - {passed_synth}/{total_synth} passed, reward={reward:.3f}")
                
                attempt_pbar.set_postfix({"synth_passed": f"{passed_synth}/{total_synth}", "reward": f"{reward:.2f}", "best": f"{best_synth_passed}/{total_synth}"})
                print(f"  Attempt {attempt+1}: {passed_synth}/{total_synth} synthetic tests passed, reward={reward:.3f}")
                
                # Check if all tests passed - if so, stop early
                if all(new_synth_results):
                    print(f"  ✓ All {len(new_synth_results)} synthetic tests passed. Stopping repair attempts.")
                    current_metadata = {}
                    break
                
                # Update metadata for next iteration
                if not all(new_synth_results):
                    try:
                        fail_index = new_synth_results.index(False)
                        failing_test = synthetic_tests[fail_index]
                        failing_metadata = new_synth_metadata["test_results"][fail_index]
                        current_metadata = {
                            "inputs": failing_test.input_val, "actual": failing_metadata.get("actual", ""),
                            "expected": failing_test.expected_output, "error_code": -2
                        }
                    except (ValueError, IndexError, KeyError):
                        current_metadata = {}

                current_code = repaired_code
                initial_synth_results = new_synth_results
                current_synth_metadata = new_synth_metadata  # Update metadata for next iteration
            # --- End of RL Repair Loop ---
            
            # Use the best code found across all attempts
            final_model_output = best_model_output
            print(f"\n[BEST CODE SELECTED] Using code from best attempt: {best_synth_passed}/{len(synthetic_tests)} synthetic tests passed, reward={best_reward:.3f}")

            # Step 5: Final diagnostic evaluation on ground-truth tests
            print(f"\n[FINAL EVALUATION] Running ground-truth evaluation for {sample.question_id}")
            final_code = extract_code(final_model_output, self.model.model_style)
            
            # Debug: Check if code extraction was successful
            if not final_code or len(final_code.strip()) == 0:
                logging.error(f"Code extraction returned empty code! Original output length: {len(final_model_output)}")
                logging.error(f"First 200 chars of output: {final_model_output[:200]}")
            
            final_eval_args = ([final_code], sample.get_evaluation_sample(), self.args.debug, self.args.timeout)
            final_gt_results, _ = evaluate_generations_by_problem(final_eval_args)
            
            # Convert error codes to True/False: True (1) means pass, anything else (-2, -3, -4, etc.) means fail
            gt_passed_list = [1 if r == 1 else 0 for r in final_gt_results[0]]
            passed_gt = sum(gt_passed_list)
            total_gt = len(final_gt_results[0])
            last_synth_results = f"{sum(new_synth_results)}/{len(new_synth_results)}"
            print(f"[FINAL EVALUATION] Ground-truth: {passed_gt}/{total_gt}, Synthetic (last): {last_synth_results}\n")

            return final_model_output
        finally:
            # This ensures the policy is updated and memory is cleared at the end of every episode
            self.rl_agent.update_policy()

    def _determine_code_type(self, sample) -> str:
        """
        Determine if the code is stdin-based or call-based.
        Checks for function_name in metadata as a heuristic.
        """
        # Check if there's a function_name in the sample metadata
        if hasattr(sample, 'metadata') and sample.metadata:
            if isinstance(sample.metadata, str):
                try:
                    metadata = json.loads(sample.metadata)
                    func_name = metadata.get('func_name') or metadata.get('fn_name')
                    if func_name and func_name != "None":
                        print(f"[DEBUG] Problem {sample.question_id}: Detected call_based with func_name='{func_name}'")
                        return "call_based"
                except:
                    pass
            elif isinstance(sample.metadata, dict):
                func_name = sample.metadata.get('func_name') or sample.metadata.get('fn_name')
                if func_name and func_name != "None":
                    print(f"[DEBUG] Problem {sample.question_id}: Detected call_based with func_name='{func_name}'")
                    return "call_based"
        
        # Default to stdin
        print(f"[DEBUG] Problem {sample.question_id}: Defaulting to stdin")
        return "stdin"
    
    def _extract_internal_diagnostics(self, metadata: dict) -> dict:
        """
        Extract a compact summary of internal test failures for state.
        """
        diagnostics = {
            "failed_categories": [],
            "error_snippets": [],
            "timeout_count": 0
        }
        
        if not isinstance(metadata, dict):
            return diagnostics
        
        # Extract failed test categories
        test_results = metadata.get("test_results", [])
        for result in test_results:
            if not result.get("passed", True):
                cat = result.get("category", "unknown")
                if cat not in diagnostics["failed_categories"]:
                    diagnostics["failed_categories"].append(cat)
                
                # Extract error snippet if available
                test_meta = result.get("metadata", {})
                error = test_meta.get("error", "")
                if error and len(error) > 0:
                    # Truncate error to first 100 chars
                    error_snippet = error[:100] if len(error) > 100 else error
                    if error_snippet not in diagnostics["error_snippets"]:
                        diagnostics["error_snippets"].append(error_snippet)
        
        # Count timeouts
        diagnostics["timeout_count"] = metadata.get("timeouts", 0)
        
        return diagnostics