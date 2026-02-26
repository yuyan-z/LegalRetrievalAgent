
# Start Repo

# 1. Installation

## Miniconda
Clone the repository.

Windows users: Install Visual Studio Build Tools or w64devkit before installing llama-cpp-python. 

```bash
# Create and activate conda environment
conda create -n py312 python=3.12.3
conda activate py312

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing/linting

# Package the project
pip install -e .  # for Kaggle submissions
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
