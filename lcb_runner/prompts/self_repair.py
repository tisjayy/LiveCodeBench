import json

from anthropic import HUMAN_PROMPT, AI_PROMPT

from lcb_runner.lm_styles import LMStyle


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. Fix the buggy code to pass all test cases. Output ONLY the complete corrected code in a code block. Do NOT include explanations, comments about what you changed, or any text outside the code block."
    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you are helping a user correct an error program for code competition. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the entire executable program. You must put the entire fixed executable program within code delimiters."
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
    Now supports 10 distinct repair strategies with clearer, more actionable prompts.
    """
    # Original 5 strategies - ALL NOW USE METADATA
    
    # Helper function to extract metadata once
    def _extract_test_info(metadata):
        """Extract test failure information from metadata."""
        try:
            meta_json = json.loads(metadata) if isinstance(metadata, str) else metadata
            inputs = meta_json.get("inputs", "N/A")
            output = meta_json.get("output", "N/A")
            expected = meta_json.get("expected", "N/A")
            error = meta_json.get("error", "")
            category = meta_json.get("test_category", "")
            total_failed = meta_json.get("total_failed", 1)
            failed_cats = meta_json.get("failed_categories", [])
            
            # Build the failed test message
            bug_msg = f"\n🐛 FAILED TEST ({category} case):\nInput:        {inputs}\nYour Output:  {output}\nExpected:     {expected}"
            if error:
                bug_msg += f"\nError:        {error}"
            
            # Add multi-test context
            if total_failed > 1:
                bug_msg += f"\n\n⚠️  Total failures: {total_failed} tests"
                if failed_cats:
                    bug_msg += f" (categories: {', '.join(failed_cats)})"
            
            return bug_msg
        except (json.JSONDecodeError, AttributeError, KeyError):
            return ""
    
    if strategy == "step_by_step_reasoning":
        test_info = _extract_test_info(metadata)
        return f"{test_info}\n\nDebug systematically: (1) Trace with the failed input above, (2) Find where output diverges from expected, (3) Fix the bug. Output corrected code ONLY."
    
    if strategy == "error_specific_feedback":
        test_info = _extract_test_info(metadata)
        return f"{test_info}\n\nFix the bug that causes this test to fail. Output corrected code ONLY (no explanations)."

    if strategy == "syntax_check":
        test_info = _extract_test_info(metadata)
        return f"{test_info}\n\nFix syntax errors: missing colons, unmatched brackets, bad indentation, undefined variables. Output corrected code ONLY."

    if strategy == "edge_case_check":
        test_info = _extract_test_info(metadata)
        return f"{test_info}\n\nFix edge case handling: Check empty inputs, boundary values (0, -1, max), off-by-one errors, special constraints. Output corrected code ONLY."

    # New 5 strategies - ALL NOW USE METADATA
    if strategy == "algorithm_optimization":
        test_info = _extract_test_info(metadata)
        return f"{test_info}\n\nAlgorithm likely wrong/inefficient. Consider: two-pointer, sliding window, DP, greedy? Using right approach? Fix and output code ONLY."
    
    if strategy == "data_structure_change":
        test_info = _extract_test_info(metadata)
        return f"{test_info}\n\nTry different data structure: dict/set for O(1) lookup, heapq for min/max, sort first if order matters. Output corrected code ONLY."
    
    if strategy == "logic_decomposition":
        test_info = _extract_test_info(metadata)
        return f"{test_info}\n\nSimplify logic: Break complex conditions, reduce nested loops/ifs, fix boolean and/or errors, handle all cases. Output corrected code ONLY."
    
    if strategy == "variable_state_tracking":
        test_info = _extract_test_info(metadata)
        return f"{test_info}\n\nFix variable/state: Initialize before use, update counters correctly, reset flags. Common: i vs i+1, forgot sum=0. Output code ONLY."
    
    if strategy == "type_and_conversion":
        test_info = _extract_test_info(metadata)
        return f"{test_info}\n\nFix type errors: Convert input() to int/float, don't compare str with int, fix indexing. Common: '123' as string not int. Output code ONLY."

    # "default" strategy - IMPROVED (also uses metadata)
    test_info = _extract_test_info(metadata)
    return f"{test_info}\n\nIdentify and fix the bug (logic error, wrong calculation, missing case). Output complete corrected code ONLY (no explanations)."


def get_check_prompt(question: str, result, metadata):
    """Format error information clearly and concisely."""
    try:
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    except (json.JSONDecodeError, TypeError):
        return ""

    if "error_code" not in metadata:
        return ""
    
    error_code = metadata["error_code"]
    
    if error_code == -1:
        # Compilation/Syntax Error
        message = f"❌ COMPILATION ERROR:\n{metadata.get('error', 'Unknown syntax error')}"
    
    elif error_code == -2:
        # Wrong Answer
        inp = metadata.get('inputs', 'N/A')
        out = metadata.get('output', 'N/A')
        exp = metadata.get('expected', 'N/A')
        message = f"❌ WRONG ANSWER:\nInput:    {inp}\nYour Output:  {out}\nExpected:     {exp}"
    
    elif error_code == -3:
        # Time Limit Exceeded
        inp = metadata.get('inputs', 'N/A')
        exp = metadata.get('expected', 'N/A')
        message = f"❌ TIME LIMIT EXCEEDED:\nInput:    {inp}\nExpected: {exp}\nYour code is too slow. Optimize the algorithm."
    
    elif error_code == -4:
        # Runtime Error
        inp = metadata.get('inputs', 'N/A')
        exp = metadata.get('expected', 'N/A')
        err = metadata.get('error', 'Unknown runtime error')
        message = f"❌ RUNTIME ERROR:\nInput:    {inp}\nExpected: {exp}\nError:    {err}"
    
    else:
        message = f"❌ ERROR (code {error_code}): {metadata.get('error', 'Unknown error')}"
    
    return message


# --- 2. UPDATED: Cleaner, more focused template ---
def get_generic_question_template_answer(question: str, code, result, metadata, strategy: str):
    # Problem specification
    prompt = f"# Problem:\n{question}\n\n"
    
    # Show the buggy code
    prompt += f"# Your Current Code (BUGGY):\n```python\n{code}\n```\n\n"
    
    # Show the error
    error_msg = get_check_prompt(question, result, metadata)
    if error_msg:
        prompt += f"{error_msg}\n\n"
    
    # Strategy-specific instruction
    strategy_instruction = get_strategy_instruction(strategy, metadata)
    prompt += f"# Task:\n{strategy_instruction}\n\n"
    
    # Request fixed code - EMPHASIZE CODE ONLY
    prompt += "# Fixed Code (CODE ONLY, NO EXPLANATIONS):\n```python\n"
    
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


def extract_code_with_llm_fallback(model_output: str, lmstyle: LMStyle, base_runner=None):
    """
    Extract code with LLM fallback for ambiguous cases.
    
    Strategy:
    1. Try ground-truth extraction (matches baseline evaluator)
    2. If empty, try enhanced pattern matching
    3. If result looks suspicious and base_runner available, use LLM to parse (slow, costs API call)
    4. Final fallback: return cleaned full output
    """
    import re
    from lcb_runner.utils.extraction_utils import extract_code as extract_code_gt
    
    # Strategy 1: Use ground-truth extraction first (matches baseline evaluator)
    code = extract_code_gt(model_output, lmstyle)
    
    # If ground-truth extraction returned empty, try fallback strategies
    if not code or code.strip() == "":
        # Strategy 2: Try to find Python code patterns
        code_patterns = [
            r'((?:^|\n)(?:import|from|class|def|if __name__|#).*?)(?:\n\n|\Z)',  # Python code blocks
            r'```(?:python)?\n(.*?)```',  # Markdown code blocks
            r'```(?:python)?\n(.*)',  # Unclosed markdown blocks
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, model_output, re.DOTALL | re.MULTILINE)
            if matches:
                # Take the longest match (most complete)
                code = max(matches, key=len).strip()
                break
        
        # Strategy 3: If still no code, use full output as last resort
        if not code or code.strip() == "":
            code = model_output.strip()
    
    # Strategy 4: Clean up common artifacts
    code = _clean_code_extraction(code)
    
    # Strategy 5: Check if extraction looks suspicious and needs LLM help
    if base_runner is not None and _needs_llm_parsing(code, model_output):
        try:
            improved_code = _llm_parse_code(model_output, base_runner)
            if improved_code and len(improved_code.strip()) > 10:  # Valid result
                code = improved_code
        except Exception as e:
            # LLM parsing failed, keep regex-extracted code
            import logging
            logging.warning(f"LLM fallback parsing failed, using regex extraction: {str(e)[:100]}")
    
    return code


def _clean_code_extraction(code: str) -> str:
    """Clean up common artifacts from code extraction."""
    import re
    
    # Remove leading/trailing explanations (common patterns)
    # Remove "Here's the code:" or similar prefixes
    code = re.sub(r'^(?:Here\'?s?|Below is|This is).*?(?:code|solution)[:\s]*\n', '', code, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove trailing explanations after code
    code = re.sub(r'\n+(?:This |The |I )(solution|code|implementation).*$', '', code, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove multiple blank lines
    code = re.sub(r'\n{3,}', '\n\n', code)
    
    # Remove markdown artifacts if they slipped through
    code = re.sub(r'^```(?:python)?\s*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
    
    return code.strip()


def _needs_llm_parsing(extracted_code: str, full_output: str) -> bool:
    """Heuristics to detect if LLM fallback parsing is needed (more conservative)."""
    # If extraction is empty or extremely short, need LLM help
    if len(extracted_code.strip()) < 30:  # Very short threshold
        return True
    
    # If code has obvious syntax errors at the end (incomplete)
    stripped = extracted_code.strip()
    if stripped.endswith(('(', '[', '{')) and not stripped.endswith(('{}', '[]', '()')):
        return True
    
    # If extracted code starts with natural language (not code)
    first_line = stripped.split('\n')[0].strip()
    if first_line and not any(first_line.startswith(x) for x in ['import', 'from', 'def', 'class', '#', '@', 'if', 'for', 'while', 'try', 'with']):
        # Check if first line looks like English prose
        if len(first_line.split()) > 5 and not any(c in first_line for c in ['=', '(', ')', '[', ']', '{', '}']):
            return True
    
    # If extracted code has excessive natural language mixed in
    lines = stripped.split('\n')
    prose_lines = 0
    for line in lines[:10]:  # Check first 10 lines only
        line_stripped = line.strip()
        if line_stripped and not line_stripped.startswith('#'):
            # Count lines that look like English prose (no code symbols)
            if len(line_stripped.split()) > 5 and not any(c in line_stripped for c in ['=', '(', ')', '[', ']', '{', '}', ':', 'def', 'class', 'import']):
                prose_lines += 1
    
    if prose_lines > 3:  # More than 3 prose-like lines in first 10
        return True
    
    return False


def _llm_parse_code(model_output: str, base_runner) -> str:
    """Use LLM to parse code from ambiguous output with retry logic."""
    import re
    
    # Truncate very long outputs to avoid token limits
    max_chars = 4000
    truncated_output = model_output[:max_chars] + ("..." if len(model_output) > max_chars else "")
    
    parse_prompt = f"""Extract ONLY the Python code from the following response. 
Do not include any explanations, markdown formatting, or commentary.
Return just the executable Python code, nothing else.
If there are multiple code blocks, return the final/corrected version.

Response:
{truncated_output}

Python code only:"""
    
    try:
        # Try to get LLM to parse it
        outputs = base_runner.prompts_to_outputs([parse_prompt])
        if not outputs or not outputs[0]:
            raise ValueError("Empty output from LLM parser")
        
        parsed = outputs[0][0].strip()
        
        if not parsed:
            raise ValueError("LLM returned empty string")
        
        # Remove markdown if LLM added it
        if "```" in parsed:
            # Find all code blocks
            code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', parsed, re.DOTALL)
            if code_blocks:
                # Take the longest/last code block
                parsed = code_blocks[-1].strip()
            else:
                # Remove partial markdown
                parsed = re.sub(r'```(?:python)?\s*\n?', '', parsed)
                parsed = re.sub(r'\n?```\s*', '', parsed)
        
        # Final validation: should look like Python code
        parsed = parsed.strip()
        if len(parsed) < 10:
            raise ValueError(f"Parsed code too short ({len(parsed)} chars)")
        
        # Should have at least one Python keyword or structure
        if not any(keyword in parsed for keyword in ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'return']):
            raise ValueError("Parsed result doesn't look like Python code")
        
        return parsed
        
    except Exception as e:
        # Provide more specific error for debugging
        error_msg = f"LLM parsing failed: {type(e).__name__}: {str(e)[:100]}"
        raise Exception(error_msg)


def extract_code(model_output: str, lmstyle: LMStyle):
    """Legacy function for compatibility - no LLM fallback."""
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    
    # Handle different cases
    if len(indexlines) >= 2:
        # Normal case: opening and closing ```
        # Use LAST pair of markers (matches ground-truth evaluator)
        return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
    elif len(indexlines) == 1:
        # Only opening ```, take everything after
        return "\n".join(outputlines[indexlines[0] + 1:])
    else:
        # No code blocks found, return full output (fallback)
        return model_output.strip()

# ... (test function remains the same) ...