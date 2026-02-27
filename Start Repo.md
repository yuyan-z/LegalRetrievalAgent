
# Start Repo

# 1. Installation

## Miniconda
Clone the repository.  

Setup environment.  
Windows users: Visual Studio Build Tools or w64devkit is required when building `llama-cpp-python` from source.  

```bash
# Create and activate conda environment
conda create -n py312 python=3.12.3
conda activate py312

# Install dependencies
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt  # for testing/linting

# Package the project
pip install -e .
```

(Optional) Enable CUDA support for `llama-cpp-python`. 

```bash
# Uninstall existing version
python -m pip uninstall -y llama-cpp-python

# Install CUDA support version (<cuda_version> could be cu121, cu122, cu124, etc.)
python -m pip install --force-reinstall --no-cache-dir --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda_version> llama-cpp-python

# Verify that CUDA support is enabled
python -c "from llama_cpp import llama_cpp; llama_cpp.llama_print_system_info()"
```

## Docker
TODO

## Kaggle
Running the LLMs locally without GPU support can be very slow. Consider using the Kaggle platform for GPU acceleration.  
TODO

# 2. Run Tests
```bash
pytest tests  # run all tests
# pytest tests/test_llm   # run tests in a directory
# pytest tests/test_llm/test_loader.py  # run a test file
```

# 3. Download Data
## Direct download
Open the [Kaggle competition data page](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/data).

Click the "Download All" button.

Extract data files into `data/` directory. 

### Using Kaggle API
TODO

# 4. Run Baselines
Place a GGUF model file (compatible with llama.cpp) in the `model/` directory.  

We can download models from the [Hugging Face llama.cpp collection](https://huggingface.co/models?apps=llama.cpp&sort=likes).
For example, [mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF?show_file_info=mistral-7b-instruct-v0.2.Q4_K_M.gguf).  
To do this, create a [Hugging Face API token](https://huggingface.co/settings/tokens), add it to `.env` file, and run `utils/download_model.py`.

```bash
python utils/download_model.py
```

Run baselines  `notebooks/01_direct_generation_baseline.ipynb` and `notebooks/02_agentic_retrieval_baseline.ipynb`. 
