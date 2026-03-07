# lexcite

# Installation

## Virtual environment
1. Clone the repository.  

2. Setup environment.  

```bash
# Create and activate conda environment
conda create -n lexcite python=3.12
conda activate lexcite

# Install dependencies
python -m pip install -r requirements.txt

# Package the project
pip install -e .
```

3. Install `llama-cpp-python`. 

This project supports multiple LLM providers. By default it uses API-based models via `langchain_openai.ChatOpenAI`, but it can also run local models using `langchain_community.llms.LlamaCpp`, which requires `llama-cpp-python`.  

Windows users: Visual Studio Build Tools or w64devkit is required when building `llama-cpp-python` from source.  
```bash
python -m pip install "llama-cpp-python>=0.3.4"
```

(Optional) Enable CUDA support for `llama-cpp-python`. 

```bash
# Install CUDA support version (<cuda_version> could be cu121, cu122, cu124, etc.)
python -m pip install --force-reinstall --no-cache-dir --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda_version> llama-cpp-python

# Verify that CUDA support is enabled
python -c "from llama_cpp import llama_cpp; llama_cpp.llama_print_system_info()"
```

4. 

## Docker
1. Reopen the project in the container.

2. Run setup.sh.
```bash
bash .devcontainer/setup.sh
```

3. Run `llm_loader.py` to test if it works.


# 2. Download Data
## Direct download
1. Go to the dataset page for the [LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/data) Kaggle competition.

2. Click the "Download All" button.

3. Extract data files into `data/` directory.  

## Kaggle API
1. Install `kaggle`.
```bash
python -m pip install "kaggle>=2.0.0"
```

2. Create a [Kaggle API token](https://www.kaggle.com/settings).  

Add the username and the key to `.env` file.

3. Run the script. 
```bash
python scripts/download_data.py
```

# 3. Load LLM
## Use GitHub Model API
1. Create a [GitHub Personal Access Token](https://github.com/settings/tokens) (classic).  

Add the token to `.env` file.

2. Select a model from [GitHub Models](https://models.github.ai/catalog/models).  

For example, select gpt-4.1 and pass the `model_name` parameter when loading the LLM.


## Use llama.cpp
Place a GGUF model file (compatible with llama.cpp) in the `models/` directory.  

We can download models from the [Hugging Face llama.cpp collection](https://huggingface.co/models?apps=llama.cpp&sort=likes).
For example, [mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF?show_file_info=mistral-7b-instruct-v0.2.Q4_K_M.gguf).  

1. Create a [Hugging Face API token](https://huggingface.co/settings/tokens).

Add the token to `.env` file.

run `utils/download_model.py`.

2. Run the script. 
```bash
python scripts/download_model.py
```
