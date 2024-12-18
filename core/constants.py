import os

from dotenv import load_dotenv


load_dotenv()

VERSION_KEY = 61_000
# Default NETUID if not set in environment
DEFAULT_NETUID = 241

try:
    NETUID = int(os.getenv("NETUID", DEFAULT_NETUID))
except (TypeError, ValueError):
    NETUID = DEFAULT_NETUID

MINER_DOCKER_IMAGE = "weightswandering/tuning_miner:latest"
MINER_DOCKER_IMAGE_DIFFUSION = "diagonalge/diffusion_miner:latest"
VALIDATOR_DOCKER_IMAGE = "weightswandering/tuning_vali:latest"
VALIDATOR_DOCKER_IMAGE_DIFFUSION = "diagonalge/gradients_vali_diffusion:latest"

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

CONFIG_DIR = "./core/config/"
OUTPUT_DIR = "./core/outputs/"
DIFFUSION_DATASET_DIR = "./core/dataset/images"

DIFFUSION_DEFAULT_REPEATS = 10
DIFFUSION_DEFAULT_INSTANCE_PROMPT = "lora"
DIFFUSION_DEFAULT_CLASS_PROMPT = "style"

CONFIG_TEMPLATE_PATH = CONFIG_DIR + "base.yml"
CONFIG_TEMPLATE_PATH_DIFFUSION = CONFIG_DIR + "base_diffusion.toml"

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")

CUSTOM_DATASET_TYPE = "custom"
