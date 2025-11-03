import json

from anthropic import HUMAN_PROMPT, AI_PROMPT

from lcb_runner.lm_styles import LMStyle


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once."
    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you are helping a user correct a error program for code competition. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the entire executable program. You must put the entire fixed executable program within code delimiters."
    SYSTEM_MESSAGE_MAGIC = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n"
    SYSTEM_MESSAGE_WIZARD = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    SYSTEM_MESSAGE_PHIND = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. You must put the entired fixed program within code delimiters only for once., for example: 
```python 
# YOUR CODE HERE
```"""
    FORMATTING_REPEAT = f"First reason about the code providing a textual explanation of what is wrong with the code and then generate a fixed of the program enclosed code delimiters."
    FORMATTING_MESSAGE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


# --- 1. NEW: Function to define RL prompt strategies ---
def get_strategy_instruction(strategy: str, metadata: dict) -> str:
    """
    Returns a specific instruction string based on the chosen RL agent's action.
    """
    if strategy == "step_by_step_reasoning":
        return "Analyze the error and provide a step-by-step explanation of your reasoning before providing the corrected code."
    
    if strategy == "error_specific_feedback":
        try:
            # Safely load metadata
            meta_json = json.loads(metadata) if isinstance(metadata, str) else metadata
            inputs = meta_json.get("inputs")
            output = meta_json.get("output")
            expected = meta_json.get("expected")
            if inputs and expected:
                return (f"Focus on the following failed test case. "
                        f"Input: '{inputs}', "
                        f"Generated Output: '{output}', "
                        f"Expected Output: '{expected}'. "
                        f"Explain why the code failed for this specific case and then provide the fix.")
        except (json.JSONDecodeError, AttributeError):
            # Fallback if metadata is malformed
            return "Please fix the error in the code."

    if strategy == "syntax_check":
        return "The code has a syntax error. Please carefully review the code for any syntax issues, typos, or compilation errors and provide a corrected version."

    if strategy == "edge_case_check":
        return "The current code fails on some inputs. Review the logic and consider potential edge cases (e.g., empty inputs, large numbers, specific constraints) that might not be handled correctly. Provide a more robust solution."

    # "default" strategy
    return "Please fix the error in the provided code."


def get_check_prompt(question: str, result, metadata):
    ## assumes i/o examples are already truncated!
    try:
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    except (json.JSONDecodeError, TypeError):
        return "" # Cannot parse metadata, return empty prompt

    if "error_code" not in metadata:
        return ""
    if metadata["error_code"] == -1:
        message = f"The above code is incorrect and got the following compilation error.\n{metadata['error']}"
    elif metadata["error_code"] == -2:
        message = f"The above code is incorrect and got a wrong answer.\nInput: {metadata.get('inputs')}\nGenerated Output: {metadata.get('output')}\nExpected: {metadata.get('expected')}"
    elif metadata["error_code"] == -3:
        message = f"The above code is incorrect and got time limit exceeded.\n{metadata.get('error')}\nInput: {metadata.get('inputs')}\nExpected: {metadata.get('expected')}"
    elif metadata["error_code"] == -4:
        message = f"The above code is incorrect and got a runtime error.\nInput: {metadata.get('inputs')}\nExpected: {metadata.get('expected')}\n{metadata.get('error')}"
    else:
        # Don't raise an error, just return a generic message
        return f"The code failed with an unknown error code: {metadata.get('error_code')}"
    return message


# --- 2. UPDATED: Generic template now accepts the strategy ---
def get_generic_question_template_answer(question: str, code, result, metadata, strategy: str):
    prompt = f"### Question:\n{question}\n\n"
    prompt += f"### Faulty Code:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata) + "\n\n"
    
    # --- Injecting the Strategy Instruction ---
    strategy_instruction = get_strategy_instruction(strategy, metadata)
    prompt += f"### Instruction:\n{strategy_instruction}\n\n"
    
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


# Note: Other get_*_template functions should be updated similarly if they are used.
# For now, we focus on the generic one which is the most common.


# --- 3. UPDATED: Main formatting function now accepts and passes the strategy ---
def format_prompt_self_repair(
    question: str, LanguageModelStyle: LMStyle, code, result, metadata, strategy: str = "default"
) -> str:
    if result:
        # The code is accepted, no need to change anything.
        return ""

    # Most models can use the generic template, so we pass the strategy to it.
    # If a model needs a special template, that template function must also be updated to handle the `strategy` argument.
    
    if LanguageModelStyle in [LMStyle.OpenAIChat, LMStyle.DeepSeekAPI]:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata, strategy
                )
                + "\n\n"
                + PromptConstants.FORMATTING_REPEAT,
            },
        ]
        return chat_messages
    if LanguageModelStyle == LMStyle.LLaMa3:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata, strategy
                ),
            },
        ]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True, truncation=False, padding=False
        )
    elif LanguageModelStyle == LMStyle.Claude:
        prompt = f"{HUMAN_PROMPT}\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n{get_generic_question_template_answer(question, code, result, metadata, strategy).rstrip()}\n{AI_PROMPT}"
        return prompt
    elif LanguageModelStyle == LMStyle.Claude3:
        system = PromptConstants.SYSTEM_MESSAGE_GENERIC
        prompt = [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata, strategy
                ).rstrip(),
            }
        ]
        return system, prompt
    
    # ... other model styles will now also use the strategy via the generic template ...
    else:
        # Generic fallback for many models
        prompt_content = get_generic_question_template_answer(question, code, result, metadata, strategy)
        
        # This logic is simplified from your original file for clarity, but demonstrates the principle
        if LanguageModelStyle in [LMStyle.Gemini, LMStyle.StarCoderInstruct]:
             return f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{prompt_content}"
        elif LanguageModelStyle == LMStyle.CodeLLaMaInstruct:
             return f"[INST] <<SYS>>\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n<</SYS>>\n\n{prompt_content}\n[/INST]"
        else:
            # A catch-all for other models that might use this structure
            return f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{prompt_content}"


def extract_code(model_output: str, lmstyle: LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])

# ... (test function remains the same) ...