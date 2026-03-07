from huggingface_hub import hf_hub_download

from lexcite.config import HUGGINGFACE_TOKEN, MODEL_DIR


def download_hf_model(repo_id: str, filename: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    if not HUGGINGFACE_TOKEN:
        raise ValueError("No HUGGINGFACE_TOKEN in .env !")
    
    print("Downloading model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(MODEL_DIR),
        token=HUGGINGFACE_TOKEN
    )
    print(f"Model downloaded: {model_path}.")


if __name__ == "__main__":
    download_config = {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    }
    # download_config = {
    #     "repo_id": "google/gemma-7b",
    #     "filename": "gemma-7b.gguf"
    # }

    download_hf_model(**download_config)
