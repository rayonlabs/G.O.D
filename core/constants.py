import os

from dotenv import load_dotenv


load_dotenv()

VERSION_KEY = 61_000
# Default NETUID if not set in environment
DEFAULT_NETUID = 56

try:
    NETUID = int(os.getenv("NETUID", DEFAULT_NETUID))
except (TypeError, ValueError):
    NETUID = DEFAULT_NETUID

MINER_DOCKER_IMAGE = "weightswandering/tuning_miner:latest"
VALIDATOR_DOCKER_IMAGE = "weightswandering/tuning_vali:latest"

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

CONFIG_DIR = "./core/config/"
OUTPUT_DIR = "./core/outputs/"

CONFIG_TEMPLATE_PATH = CONFIG_DIR + "base.yml"

BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")

REPO_ID = os.getenv("REPO_ID")
CUSTOM_DATASET_TYPE = "custom"
