import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner


class OpenAIRunner(BaseRunner):
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs: dict[str | str] = {
                "model": model.model_name,  # Use model parameter, not args.model
                "max_completion_tokens": 25000,
            }
        elif model.model_style == LMStyle.OpenAIReason:
            assert (
                "__" in model.model_name
            ), f"Model {model.model_name} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
            model_name, reasoning_effort = model.model_name.split("__")
            self.client_kwargs: dict[str | str] = {
                "model": model_name,
                "reasoning_effort": reasoning_effort,
            }
        else:
            self.client_kwargs: dict[str | str] = {
                "model": model.model_name,  # Use model parameter, not args.model
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }

    def _run_single(self, prompt: list[dict[str, str]], n: int = 1) -> list[str]:
        assert isinstance(prompt, list)

        if n == 0:
            print("Max retries (1) reached for OpenAI API. Giving up and returning empty response.")
            return [""]  # Return list with empty string

        # Log which model is being used for this API call
        model_being_used = self.client_kwargs.get("model", "unknown")
        import logging
        logging.debug(f"OpenAIRunner._run_single: Making API call with model={model_being_used}")
        
        # Print to stderr to avoid interfering with progress bars
        import sys
        if n == 1:  # Only log on first attempt, not retries
            print(f"  [API Call] Using model: {model_being_used}", file=sys.stderr, flush=True)

        try:
            response = OpenAIRunner.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            retry_num = 2 - n  # Calculate current retry (1-indexed)
            print(f"Exception: {repr(e)} (Retry {retry_num}/1)")
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single(prompt, n=n - 1)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content for c in response.choices]
