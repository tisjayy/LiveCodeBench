from lcb_runner.lm_styles import LMStyle, LanguageModel
from lcb_runner.utils.scenarios import Scenario

# --- 1. Import our custom RL runner ---
from lcb_runner.runner.rl_self_repair_runner import RLSelfRepairRunner

def build_runner(args, model: LanguageModel):
    # --- 2. Add the RL runner selection logic at the top ---
    if args.scenario == Scenario.selfrepair and args.use_rl:
        print("INFO: Building RL Self-Repair Runner...")
        # If the scenario is selfrepair AND --use_rl is active,
        # return an instance of our custom runner. The rest of this function is skipped.
        return RLSelfRepairRunner(args, model)
    # --- End of new logic ---

    # The original logic of this function remains unchanged.
    # It will be executed if the condition above is false.
    if model.model_style == LMStyle.OpenAIChat:
        from lcb_runner.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model)
    if model.model_style in [LMStyle.OpenAIReason, LMStyle.OpenAIReasonPreview]:
        from lcb_runner.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model)
    if model.model_style in [LMStyle.Gemini, LMStyle.GeminiThinking]:
        from lcb_runner.runner.gemini_runner import GeminiRunner

        return GeminiRunner(args, model)
    if model.model_style in [LMStyle.Claude3, LMStyle.Claude3Thinking]:
        from lcb_runner.runner.claude3_runner import Claude3Runner

        return Claude3Runner(args, model)
    if model.model_style == LMStyle.Claude:
        from lcb_runner.runner.claude_runner import ClaudeRunner

        return ClaudeRunner(args, model)
    if model.model_style == LMStyle.MistralWeb:
        from lcb_runner.runner.mistral_runner import MistralRunner

        return MistralRunner(args, model)
    if model.model_style == LMStyle.CohereCommand:
        from lcb_runner.runner.cohere_runner import CohereRunner

        return CohereRunner(args, model)
    if model.model_style == LMStyle.DeepSeekAPI:
        from lcb_runner.runner.deepseek_runner import DeepSeekRunner

        return DeepSeekRunner(args, model)
    if model.model_style == LMStyle.DeepSeekAPI:
        from lcb_runner.runner.deepseek_runner import DeepSeekRunner

        return DeepSeekRunner(args, model)
    if model.model_style == LMStyle.Grok:
        from lcb_runner.runner.grok_runner import GrokRunner

        return GrokRunner(args, model)
    if model.model_style == LMStyle.TogetherAI:
        from lcb_runner.runner.together_runner import TogetherAIRunner

        return TogetherAIRunner(args, model)
    if "/fireworks/" in model.model_name:
        from lcb_runner.runner.fireworks_runner import FireWorksRunner

        return FireWorksRunner(args, model)
    elif model.model_style in []:
        raise NotImplementedError(
            f"Runner for language model style {model.model_style} not implemented yet"
        )
    else:
        from lcb_runner.runner.vllm_runner import VLLMRunner

        return VLLMRunner(args, model)