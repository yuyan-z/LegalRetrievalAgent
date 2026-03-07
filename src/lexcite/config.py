# from langchain_community.llms import LlamaCpp
# from langchain_core.messages import HumanMessage

# llm = LlamaCpp(
#     model_path="./models/qwen2.5-3b-instruct-q4_k_m.gguf",
#     n_ctx=2048,
#     n_threads=8,
#     temperature=0.7,
#     max_tokens=256,
# )

# if __name__ == "__main__":
#     while True:
#         user_input = input("你: ")
#         if user_input == "exit":
#             break

#         response = llm.invoke(user_input)

#         print("AI:", response)
import os

from pathlib import Path
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT_DIR / "output"

load_dotenv(ROOT_DIR / ".env")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL_API = os.getenv("GITHUB_MODEL_API")
