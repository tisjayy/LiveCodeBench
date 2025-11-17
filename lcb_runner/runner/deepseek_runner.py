import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


class DeepSeekRunner(BaseRunner):
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API"),
        base_url="https://api.deepseek.com",
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        self.client_kwargs: dict[str | str] = {
            "model": model.model_name,  # Use model parameter, not args.model
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "timeout": args.openai_timeout,
            # "stop": args.stop, --> stop is only used for base models currently
        }

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list)

        # Log which model is being used for this API call
        model_being_used = self.client_kwargs.get("model", "unknown")
        import logging
        import sys
        logging.debug(f"DeepSeekRunner._run_single: Making API call with model={model_being_used}")
        print(f"  [DeepSeek API Call] Using model: {model_being_used}", file=sys.stderr, flush=True)

        def __run_single(counter, retry_count=0, max_retries=5):
            try:
                response = self.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
                content = response.choices[0].message.content
                return content
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
                if retry_count >= max_retries:
                    print(f"Max retries ({max_retries}) reached. Giving up.")
                    return ""  # Return empty string instead of infinite loop
                
                print(f"Exception: {repr(e)} (Retry {retry_count + 1}/{max_retries})")
                print("Sleeping for 30 seconds...")
                print("Consider reducing the number of parallel processes.")
                sleep(30)
                return __run_single(counter, retry_count + 1, max_retries)
            except Exception as e:
                print(f"Failed to run the model for {prompt}!")
                print("Exception: ", repr(e))
                raise e

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(__run_single(10))
        except Exception as e:
            raise e
        return outputs
