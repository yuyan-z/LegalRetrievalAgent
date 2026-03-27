from typing import Any


def load_chat_model(provider: str, **kwargs) -> Any:
    if provider == "llama_cpp":
        from langchain_community.chat_models import ChatLlamaCpp

        return ChatLlamaCpp(**kwargs)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**kwargs)
    
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(**kwargs)

    raise ValueError(f"Unsupported provider: {provider}")


if __name__ == "__main__":
    from uretriever.configs.path_config import MODEL_DIR
    from uretriever.configs.api_config import get_env


    # model = load_chat_model(
    #     provider="openai",
    #     model="gpt-4.1",
    #     base_url=get_env("GITHUB_MODEL_API"),
    #     api_key=get_env("GITHUB_TOKEN"),
    #     temperature=0,
    #     max_tokens=512,
    # )
    model = load_chat_model(
        provider="llama_cpp",
        model_path=str(MODEL_DIR / "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"),
        temperature=0,
        max_tokens=512,
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=-1,
        verbose=False,
    )
    # model = load_chat_model(
    #     provider="ollama",
    #     model="mistral:7b-instruct-v0.2-q4_K_M",
    #     temperature=0,
    #     num_predict=512,
    #     num_ctx=4096,
    #     verbose=False,
    # )

    response = model.invoke("What is LangChain in one sentence?")
    print("AI Response:", response.content)
