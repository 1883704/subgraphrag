import argparse
import json
import os
from pathlib import Path

from openai import OpenAI


def load_config(config_path):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)

        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required. Install it with `pip install pyyaml`.") from exc
        return yaml.safe_load(f) or {}


def flatten_config(config):
    result = dict(config)

    parameters = result.pop("parameters", None)
    if isinstance(parameters, dict):
        for key, value in parameters.items():
            if isinstance(value, dict) and "value" in value:
                result[key] = value["value"]
            else:
                result[key] = value

    api_config = result.pop("api", None)
    if isinstance(api_config, dict):
        if "key" in api_config:
            result["api_key"] = api_config["key"]
        if "key_env" in api_config:
            result["api_key_env"] = api_config["key_env"]
        if "base_url" in api_config:
            result["api_base_url"] = api_config["base_url"]

    if "base_url" in result and "api_base_url" not in result:
        result["api_base_url"] = result["base_url"]
    if "key" in result and "api_key" not in result:
        result["api_key"] = result["key"]
    if "key_env" in result and "api_key_env" not in result:
        result["api_key_env"] = result["key_env"]

    return result


def redact(value):
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "..." + value[-4:]


def main():
    parser = argparse.ArgumentParser(description="Test an OpenAI-compatible API config.")
    parser.add_argument("--config", default="configs/api.local.yaml", help="YAML/JSON config path.")
    parser.add_argument("--message", default="Reply with exactly: ok", help="Test message.")
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()

    config = flatten_config(load_config(args.config))
    model_name = config.get("model_name")
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    api_key = config.get("api_key") or os.getenv(api_key_env)
    api_base_url = config.get("api_base_url") or os.getenv("OPENAI_BASE_URL")

    print("Resolved API config:")
    print(f"  config: {args.config}")
    print(f"  model_name: {model_name or '<missing>'}")
    print(f"  api_base_url: {api_base_url or '<missing>'}")
    print(f"  api_key_env: {api_key_env}")
    print(f"  api_key: {redact(api_key)}")

    if not model_name:
        raise ValueError("model_name is missing in config.")
    if not api_key:
        raise ValueError(f"API key is missing. Set api.key in config or export {api_key_env}.")
    if not api_base_url:
        raise ValueError("api.base_url is missing in config or OPENAI_BASE_URL.")
    if api_base_url.rstrip("/").endswith("/chat/completions"):
        raise ValueError("api.base_url should not include /chat/completions. Use the provider base URL ending at /v1.")

    client = OpenAI(api_key=api_key, base_url=api_base_url)
    print("\nSending test request...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": args.message}],
        max_tokens=args.max_tokens,
        temperature=0,
    )

    print("\nSuccess. Response:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
