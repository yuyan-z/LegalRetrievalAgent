from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
PROMPT_DIR = ROOT_DIR / "prompts"
OUTPUT_DIR = ROOT_DIR / "outputs"
