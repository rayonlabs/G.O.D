import os

from core.constants import NETUID


SUCCESS = "success"
ACCOUNT_ID = "account_id"
MESSAGE = "message"
AMOUNT = "amount"
UNDELEGATION = "undelegation"
STAKE = "stake"
VERIFIED = "verified"
REDIS_KEY_COLDKEY_STAKE = "coldkey_stake"
API_KEY = "api_key"
COLDKEY = "coldkey"


BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
DELETE_S3_AFTER_COMPLETE = True

VALI_CONFIG_PATH = "validator/test_axolotl.yml"

# db stuff
NULL_ACCOUNT_ID = "00000000-0000-0000-0000-000000000000"


# api stuff should move this out to be shared by both miner and vali code?
START_TRAINING_ENDPOINT = "/start_training/"
TASK_OFFER_ENDPOINT = "/task_offer/"
SUBMISSION_ENDPOINT = "/get_latest_model_submission/"

# TODO update when live
DEV_CONTENT_BASE_URL = "https://dev.content.gradients.io"
PROD_CONTENT_BASE_URL = "https://dev.content.gradients.io"


# 241 is testnet
CONTENT_BASE_URL = DEV_CONTENT_BASE_URL if NETUID == 241 else PROD_CONTENT_BASE_URL

GET_RANDOM_DATASETS_ENDPOINT = f"{CONTENT_BASE_URL}/datasets/random"
GET_RANDOM_MODELS_ENDPOINT = f"{CONTENT_BASE_URL}/models/random"
GET_COLUMNS_FOR_DATASET_ENDPOINT = f"{CONTENT_BASE_URL}/dataset/{{dataset}}/columns/suggest"


GET_ALL_DATASETS_ID = "dataset_id"
GET_ALL_MODELS_ID = "model_id"


HOW_MANY_TASKS_ALLOWED_AT_ONCE = 15
NUMBER_OF_MINUTES_BETWEEN_SYNTH_TASK_CHECK = 15


# data stuff
TEST_SIZE = 0.1
TRAIN_TEST_SPLIT_PERCENTAGE = 0.1
GET_SYNTH_DATA = True
MAX_SYNTH_DATA_POINTS = 100
ADDITIONAL_SYNTH_DATA_PERCENTAGE = 1.0  # same size as training set
IMAGE_TRAIN_SPLIT_ZIP_NAME = "train_data.zip"
IMAGE_TEST_SPLIT_ZIP_NAME = "test_data.zip"
TEMP_PATH_FOR_IMAGES = "/tmp/validator/temp_images"

# synth stuff
SYNTH_GEN_BATCH_SIZE = 10
SYNTH_MODEL_TEMPERATURE = 0.4
CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"
_gpu_ids = os.getenv("GPU_IDS", "").strip()
GPU_IDS = [int(id) for id in _gpu_ids.split(",")] if _gpu_ids else [0]


SYNTH_MODEL = "chat-llama-3-2-3b"
PROMPT_GEN_ENDPOINT = "https://api.nineteen.ai/v1/chat/completions"
IMAGE_GEN_ENDPOINT = "https://api.nineteen.ai/v1/text-to-image"
GRADIENTS_ENDPOINT = "https://api.gradients.io/validator-signup"
PROMPT_PATH = "validator/prompts.yml"
NINETEEN_API_KEY = os.getenv("NINETEEN_API_KEY")
# Probability for using output reformulation method
OUTPUT_REFORMULATION_PROBABILITY = 0.5

# Task Stuff
MINIMUM_MINER_POOL = 1


MIN_IDEAL_NUM_MINERS_IN_POOL = 7
MAX_IDEAL_NUM_MINERS_IN_POOL = 14
MIN_TEXT_COMPETITION_HOURS = 1
MAX_TEXT_COMPETITION_HOURS = 5
MIN_IMAGE_COMPETITION_HOURS = 1
MAX_IMAGE_COMPETITION_HOURS = 2
TASK_TIME_DELAY = 15  # number of minutes we wait to retry an organic request
# how many times in total do we attempt to delay an organic request looking for miners
MAX_DELAY_TIMES = 6


# scoring stuff
SOFTMAX_TEMPERATURE = 0.5
TEST_SCORE_WEIGHTING = 0.7  # synth will be (1 - this)
TARGET_SCORE_RATIO = 1.05
MIN_TASK_SCORE = 0.0  # very tiny punishment while miners find their feet
MAX_TASK_SCORE = 1.6
TASK_SCORE_THRESHOLD = 0.9
REWEIGHTING_EXP = 0.6  # how much of a drop off from leader
SCORING_WINDOW = 7  # number of days over which we score

# processing stuff
MAX_CONCURRENT_MINER_ASSIGNMENTS = 5
MAX_CONCURRENT_TASK_PREPS = 3
MAX_CONCURRENT_TRAININGS = 10
MAX_CONCURRENT_EVALUATIONS = 1
MAX_TIME_DELAY_TO_FIND_MINERS = 1  # hours
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_TEXT = 0.8  # image is currently 1 minus this

# diffusion eval stuff
LORA_WORKFLOW_PATH = "validator/evaluation/comfy_workflows/lora.json"
CHECKPOINTS_SAVE_PATH = "validator/evaluation/ComfyUI/models/checkpoints"
LORAS_SAVE_PATH = "validator/evaluation/ComfyUI/models/loras"
DEFAULT_STEPS = 10
DEFAULT_CFG = 4
DEFAULT_DENOISE = 0.9
SUPPORTED_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
DIFFUSION_HF_DEFAULT_FOLDER = "checkpoint"
DIFFUSION_HF_DEFAULT_CKPT_NAME = "last.safetensors"


MAX_CONCURRENT_JOBS = 4


LOGPATH = "/root/G.O.D/validator/logs"


# Image generation parameters
IMAGE_GEN_MODEL = "black-forest-labs/FLUX.1-schnell"
IMAGE_GEN_STEPS = 8
IMAGE_GEN_CFG_SCALE = 3
IMAGE_GEN_HEIGHT = 1024
IMAGE_GEN_WIDTH = 1024
