import os
import time
import openai
from openai import OpenAI
from functools import partial
from prompts import icl_user_prompt, icl_ass_prompt


def _configure_hf_endpoint():
    endpoint = os.getenv("HF_ENDPOINT") or os.getenv("HF_HUB_URL")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HF_HUB_URL"] = endpoint
    else:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_URL", "https://hf-mirror.com")


def is_api_backend(llm_backend, model_name):
    llm_backend = (llm_backend or "auto").lower()
    return llm_backend == "api" or (llm_backend == "auto" and "gpt" in model_name.lower())


def llm_init(model_name, tensor_parallel_size=1, max_seq_len_to_capture=8192, max_tokens=4000, seed=0, temperature=0, frequency_penalty=0, top_p=1.0, presence_penalty=0.0, request_timeout=60, llm_backend="auto", api_key=None, api_key_env="OPENAI_API_KEY", api_base_url=None):
    if not is_api_backend(llm_backend, model_name):
        from vllm import LLM, SamplingParams

        _configure_hf_endpoint()
        client = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, max_seq_len_to_capture=max_seq_len_to_capture)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                         frequency_penalty=frequency_penalty)
        llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
    else:
        api_key = api_key or os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"{api_key_env} must be set when using API models. "
                f"Example: export {api_key_env}=your_api_key"
            )
        base_url = api_base_url or os.getenv("OPENAI_BASE_URL")
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        llm = partial(
            client.chat.completions.create,
            model=model_name,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            timeout=request_timeout,
        )
    return llm


def get_outputs(outputs, model_name, llm_backend="auto"):
    if not is_api_backend(llm_backend, model_name):
        return outputs[0].outputs[0].text
    else:
        return outputs.choices[0].message.content


def llm_inf(llm, prompts, mode, model_name, llm_backend="auto"):
    res = []
    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        outputs = get_outputs(llm(messages=conversation), model_name, llm_backend)
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs = get_outputs(llm(messages=conversation), model_name, llm_backend)
        res.append(outputs)
    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = get_outputs(llm(messages=conversation), model_name, llm_backend)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, llm_backend, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name, llm_backend)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            wait_time = (2 ** retries) * 5  # Exponential backoff
            print(f"{type(e).__name__} encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your API key, base URL, rate limits, or network.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, llm_backend="auto", max_retries=5):
    if is_api_backend(llm_backend, model_name):
        return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, llm_backend, max_retries)
    else:
        return llm_inf(llm, each_qa, llm_mode, model_name, llm_backend)
