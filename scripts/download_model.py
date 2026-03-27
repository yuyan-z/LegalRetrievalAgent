
from uretriever.configs.api_config import get_env
from uretriever.configs.path_config import MODEL_DIR


def download_hf_model(repo_id: str, filename: str):
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    huggingface_token = get_env("HUGGINGFACE_TOKEN")
    
    print("Downloading model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(MODEL_DIR),
        token=huggingface_token
    )
    print(f"Model downloaded: {model_path}.")


if __name__ == "__main__":
    download_config = {
        "repo_id": "QuantFactory/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
    }
    download_hf_model(**download_config)

    # download_config = {
    #     "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
    #     "filename": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    # }
    # download_hf_model(**download_config)

    # download_config = {
    #     "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
    #     "filename": "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
    # }
    # download_hf_model(**download_config)

    
