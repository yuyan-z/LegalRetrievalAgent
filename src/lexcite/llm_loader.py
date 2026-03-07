from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp

from lexcite.config import GITHUB_MODEL_API, GITHUB_TOKEN, MODEL_DIR


PROVIDER_CONFIG = {
    "chatopenai": {
        "class": ChatOpenAI,
        "required": ["model", "base_url", "api_key"],
        "optional": [
            # 0 = Prefer high-probability tokens (more deterministic output). higher = Allow to try lower-probability tokens (more diverse output).
            "temperature",
            # Maximum number of tokens allowed to generate. small = control cost.
            "max_tokens",
            # Maximum time (in seconds) to wait for the API response.
            "timeout",
            # Number of retry attempts if the API request fails.
            "max_retries",
            "streaming",  # Stream tokens incrementally as they are generated.
            # Select tokens in descending probability until the cumulative probability ≥ top_p. 1 = Keep all tokens.
            "top_p",
            "verbose"  # Verbose logging for debugging and tracing model calls.
        ]
    },
    "llama_cpp": {
        "class": LlamaCpp,
        "required": ["model_path"],
        "optional": [
            # 0 = favor high-probability tokens (more deterministic output). higher = allow to try lower-probability tokens (more diverse output).
            "temperature",
            # Maximum number of tokens allowed to generate. small = control cost.
            "max_tokens",
            "n_gpu_layers",  # -1 = offload all layers to GPU.
            "n_threads",  # CPU threads.
            "n_ctx",  # Context window. The default value is 512.
        ]
    },
}


def load_model(provider: str, **kwargs):
    provider = provider.lower()

    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Provider not in {list(PROVIDER_CONFIG.keys())} !")

    config = PROVIDER_CONFIG[provider]

    params = [param for param in config["required"] if param not in kwargs]
    if params:
        raise ValueError(
            f"Missing required parameters for {provider}: {params}")

    return config["class"](**kwargs)


if __name__ == "__main__":
    # model = load_model(
    #     provider="chatopenai",
    #     model="gpt-4.1",
    #     base_url=GITHUB_MODEL_API,
    #     api_key=GITHUB_TOKEN,
    #     temperature=0,   # Prefer high-probability tokens. Suitable for RAG tasks
    #     max_tokens=512,  # Limit response length. Suitable for our tasks
    # )
    # response = model.invoke("What is LangChain in one sentence?")
    # print("AI Response:", response.content)

    model = load_model(
        provider="llama_cpp",
        model_path=str(MODEL_DIR / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        temperature=0,   # Prefer high-probability tokens. Suitable for RAG tasks
        max_tokens=512,  # Limit response length. Suitable for our tasks
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=-1,
        verbose=False
    )
    response = model.invoke("What is LangChain in one sentence?")
    print("AI Response:", response)
