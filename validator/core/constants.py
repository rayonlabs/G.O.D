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
START_TRAINING_IMAGE_ENDPOINT = "/start_training_image/"
TASK_OFFER_ENDPOINT = "/task_offer/"
TASK_OFFER_IMAGE_ENDPOINT = "/task_offer_image/"
SUBMISSION_ENDPOINT = "/get_latest_model_submission/"

# TODO update when live
DEV_CONTENT_BASE_URL = "https://dev.content.gradients.io"
PROD_CONTENT_BASE_URL = "https://content.gradients.io"


# 241 is testnet
CONTENT_BASE_URL = DEV_CONTENT_BASE_URL if NETUID == 241 else PROD_CONTENT_BASE_URL

GET_RANDOM_DATASETS_ENDPOINT = f"{CONTENT_BASE_URL}/datasets/random"
GET_RANDOM_MODELS_ENDPOINT = f"{CONTENT_BASE_URL}/models/random"
GET_COLUMNS_FOR_DATASET_ENDPOINT = f"{CONTENT_BASE_URL}/dataset/{{dataset}}/columns/suggest"
GET_IMAGE_MODELS_ENDPOINT = f"{CONTENT_BASE_URL}/images/models"


GET_ALL_DATASETS_ID = "dataset_id"
GET_ALL_MODELS_ID = "model_id"


NUMBER_OF_MINUTES_BETWEEN_SYNTH_TASK_CHECK = 5


# data stuff
TEST_SIZE = 0.1
TRAIN_TEST_SPLIT_PERCENTAGE = 0.1
GET_SYNTH_DATA = True
MAX_SYNTH_DATA_POINTS = 300
MAX_TEST_DATA_POINTS = 800

ADDITIONAL_SYNTH_DATA_PERCENTAGE = 1.0  # same size as training set
IMAGE_TRAIN_SPLIT_ZIP_NAME = "train_data.zip"
IMAGE_TEST_SPLIT_ZIP_NAME = "test_data.zip"
TEMP_PATH_FOR_IMAGES = "/tmp/validator/temp_images"
SUPPORTED_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_FILE_SIZE_BYTES = 2147483646  # pyarrow max json load size
MINIMUM_DATASET_ROWS = 5000  # Minimum number of rows required in a dataset

# synth stuff
NUM_SYNTH_RETRIES = 3
SYNTH_GEN_BATCH_SIZE = 10
SYNTH_MODEL_TEMPERATURE = 0.4
CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"
_gpu_ids = os.getenv("GPU_IDS", "").strip()
GPU_IDS = [int(id) for id in _gpu_ids.split(",")] if _gpu_ids else [0]


# we sample datasets with these num_rows ranges equally
DATASET_BINS_TO_SAMPLE = [
    (30_000, 80_000), # we don't sample these for now as they are too small
    (80_000, 150_000),
    (150_000, 1_500_000),
]

# dataset row bins to training hours range
TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE = {
    #   (5_000, 10_000): (3, 5),  # 5k-10k rows needs 1-2 hours
    (10_000, 25_000): (3, 6),  # 10k-25k rows needs 2-4 hours
    (25_000, 50_000): (4, 8),  # 25k-50k rows needs 3-6 hours
    (50_000, 500_000): (5, 9),  # 50k-500k rows needs 4-8 hours
    (100_000, 500_000): (7, 10),  # 50k-500k rows needs 4-8 hours
    (500_000, 5_000_000): (8, 12),  # 500k+ rows needs 5-12 hours
}

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


MIN_IDEAL_NUM_MINERS_IN_POOL = 5
MAX_IDEAL_NUM_MINERS_IN_POOL = 9
MIN_TEXT_COMPETITION_HOURS = 2
MAX_TEXT_COMPETITION_HOURS = 12
MIN_IMAGE_COMPETITION_HOURS = 1
MAX_IMAGE_COMPETITION_HOURS = 2
TASK_TIME_DELAY = 15  # number of minutes we wait to retry an organic request
# how many times in total do we attempt to delay an organic request looking for miners
MAX_DELAY_TIMES = 6
# Maximum number of evaluation attempts when all scores are zero (including the first one)
MAX_EVAL_ATTEMPTS = 4


# scoring stuff  - NOTE: Will want to slowly make more exponential now we have auditing
TEST_SCORE_WEIGHTING = 0.7  # synth will be (1 - this)
SCORE_PENALTY = -0.05
FIRST_PLACE_SCORE  = 2
SECOND_PLACE_SCORE = 1

SIGMOID_STEEPNESS = 10  # Higher = sharper transition
SIGMOID_SHIFT = 0.3  # Shifts sigmoid curve horizontally
SIGMOID_POWER = 4  # Higher = more extreme difference between high and low scores
LINEAR_WEIGHT = 0.1  # Weight for linear component (0-1) - benefits low scores
SIGMOID_WEIGHT = 0.75  # Weight for sigmoid component (0-1) - benefits high scores

REWEIGHTING_EXP = 0.7  # how much of a drop off from leader

SCORING_WINDOW = 7  # number of days over which we score
OUTLIER_STD_THRESHOLD = 2.0  # number of standard deviations from the mean to reject the outlier scores

# processing stuff
MAX_CONCURRENT_MINER_ASSIGNMENTS = 5
MAX_CONCURRENT_TASK_PREPS = 3
MAX_CONCURRENT_TRAININGS = 10
MAX_CONCURRENT_EVALUATIONS = 1
MAX_TIME_DELAY_TO_FIND_MINERS = 1  # hours
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_TEXT = 0.75  # image is currently 1 minus this

# diffusion eval stuff
LORA_WORKFLOW_PATH = "validator/evaluation/comfy_workflows/lora.json"
LORA_WORKFLOW_PATH_DIFFUSERS = "validator/evaluation/comfy_workflows/lora_diffusers.json"
CHECKPOINTS_SAVE_PATH = "validator/evaluation/ComfyUI/models/checkpoints"
DIFFUSERS_PATH = "validator/evaluation/ComfyUI/models/diffusers"
LORAS_SAVE_PATH = "validator/evaluation/ComfyUI/models/loras"
DEFAULT_STEPS = 20
DEFAULT_CFG = 8
DEFAULT_DENOISE = 0.9
DIFFUSION_HF_DEFAULT_FOLDER = "checkpoint"
DIFFUSION_HF_DEFAULT_CKPT_NAME = "last.safetensors"
DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT = 0.5


# Max jobs
MAX_CONCURRENT_JOBS = 60
MAX_CONCURRENT_SYNTHETIC_JOBS = 15
## This leaves room for MAX_CONCURRENT_JOBS - MAX_CONCURRENT_SYNTHETIC_JOBS at all times


LOGPATH = "/root/G.O.D/validator/logs"


# Image generation parameters
IMAGE_GEN_MODEL = "black-forest-labs/FLUX.1-schnell"
IMAGE_GEN_STEPS = 8
IMAGE_GEN_CFG_SCALE = 3

MIN_IMAGE_SYNTH_PAIRS = 10
MAX_IMAGE_SYNTH_PAIRS = 50

MIN_IMAGE_WIDTH = 1024
MAX_IMAGE_WIDTH = 1024
MIN_IMAGE_HEIGHT = 1024
MAX_IMAGE_HEIGHT = 1024
IMAGE_RESOLUTION_STEP = 64  # Ensures we get resolutions divisible by 64

# scoring stuff
TEXT_TASK_SCORE_WEIGHT = 0.75
IMAGE_TASK_SCORE_WEIGHT = 1 - TEXT_TASK_SCORE_WEIGHT

SEVEN_DAY_SCORE_WEIGHT = 0.25
THREE_DAY_SCORE_WEIGHT = 0.4
ONE_DAY_SCORE_WEIGHT = 0.35
