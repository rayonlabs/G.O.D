import os

from core.constants import GRPO_DEFAULT_FIELD_PROMPT
from core.constants import NETUID


RAYONLABS_HF_USERNAME = "rayonlabs"

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
START_TRAINING_GRPO_ENDPOINT = "/start_training_grpo/"
TASK_OFFER_ENDPOINT = "/task_offer/"
TASK_OFFER_IMAGE_ENDPOINT = "/task_offer_image/"
SUBMISSION_ENDPOINT = "/get_latest_model_submission/"
TRAINING_REPO_ENDPOINT = "/training_repo"

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
MAX_TEST_DATA_POINTS = 1000

ADDITIONAL_SYNTH_DATA_PERCENTAGE = 1.0  # same size as training set

# Synthetic data constants - used for both DPO and Instruct Text tasks
SYNTHETIC_TOTAL_SIZE = 1200
SYNTHETIC_FOR_TRAINING = 700
SYNTHETIC_FOR_EVAL = 500
SYNTH_EXAMPLES_FROM_TRAIN = 600
IMAGE_TRAIN_SPLIT_ZIP_NAME = "train_data.zip"
IMAGE_TEST_SPLIT_ZIP_NAME = "test_data.zip"
TEMP_PATH_FOR_IMAGES = "/tmp/validator/temp_images"
SUPPORTED_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_FILE_SIZE_BYTES = 2_147_483_646  # pyarrow max json load size
MINIMUM_DATASET_ROWS = 2_000  # Minimum number of rows required in a dataset
EXAMPLE_PROMPTS_PATH = "validator/tasks/example_prompts.json"

# synth stuff
NUM_SYNTH_RETRIES = 3
SYNTH_GEN_BATCH_SIZE = 30
CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

# Multi-dataset augmentation
MIN_DATASETS_FOR_AUGMENTATION = 2
MAX_DATASETS_FOR_AUGMENTATION = 16

_gpu_ids = os.getenv("GPU_IDS", "").strip()
GPU_IDS = [int(id) for id in _gpu_ids.split(",")] if _gpu_ids else [0]
PROBABILITY_OF_A_BIG_TEXT_MODEL = 0.05

# we sample datasets with these num_rows ranges equally
DATASET_BINS_TO_SAMPLE = [
    (20_000, 50_000),
    (50_000, 100_000),
    (100_000, 500_000),
]

# dataset row bins to training hours range
INSTRUCT_TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE = {
    (1_000, 10_000): (1, 3),  # 1k-10k rows needs 1-3 hours
    (10_000, 25_000): (3, 6),  # 10k-25k rows needs 2-4 hours
    (25_000, 50_000): (4, 8),  # 25k-50k rows needs 3-6 hours
    (50_000, 100_000): (5, 9),  # 50k-500k rows needs 4-8 hours
    (100_000, 500_000): (7, 10),  # 50k-500k rows needs 4-8 hours
}

# text augmentation synth
TEXT_SYNTH_MODEL = "casperhansen/deepseek-r1-distill-qwen-32b-awq"
TEXT_SYNTH_WEAKER_MODEL = "llama-3-2-3b"
TEXT_SYNTH_MODEL_TEMPERATURE = 0.6
TEXT_SYNTH_MODEL_MAX_TOKENS = 5024
END_OF_REASONING_TAG = "</think>"

# image prompt generation synth
IMAGE_PROMPT_GEN_MODEL = "casperhansen/deepseek-r1-distill-qwen-32b-awq"
IMAGE_PROMPT_GEN_MODEL_TEMPERATURE = 0.4
IMAGE_PROMPT_GEN_MODEL_MAX_TOKENS = 5024
IMAGE_STYLE_PICKING_NUM_TRIES = 10
PERSON_GEN_RETRIES = 3

# endpoints
PROMPT_GEN_ENDPOINT = "https://api.nineteen.ai/v1/chat/completions"
IMAGE_GEN_ENDPOINT = "https://api.nineteen.ai/v1/text-to-image"
GRADIENTS_ENDPOINT = "https://api.gradients.io/validator-signup"
PROMPT_PATH = "validator/prompts.yml"
NINETEEN_API_KEY = os.getenv("NINETEEN_API_KEY")
EMISSION_BURN_HOTKEY = "5GU4Xkd3dCGTU3s8VLcHGc5wsD5M8XyxDca5yDQhYm1mVXFu"


# Task Stuff
MINIMUM_MINER_POOL = 1

# Tournament Start Requirements
MIN_MINERS_FOR_TOURN = 8

# Tournament Selection by Stake
TOURNAMENT_TOP_N_BY_STAKE = 32
TOURNAMENT_REPEAT_BOOST_PERCENTAGE = 5  # 5% boost per previous entry
TOURNAMENT_MAX_REPEAT_BOOST_PERCENTAGE = 25  # Maximum 25% boost
TOURNAMENT_PARTICIPATION_WEIGHT = 0.0001  # Weight given to active participants

# Tournament weight distribution
TOURNAMENT_WINNER_MIN_WEIGHT = 0.5  # Minimum weight proportion for tournament winner
TOURNAMENT_WEIGHT_DECAY_RATE = 8.0  # Controls exponential decay for non-winners


# General miner pool sizes
MIN_IDEAL_NUM_MINERS_IN_POOL = 8
MAX_IDEAL_NUM_MINERS_IN_POOL = 15

# Image-specific miner pool sizes
MIN_IDEAL_NUM_MINERS_IN_IMAGE_POOL = 15
MAX_IDEAL_NUM_MINERS_IN_IMAGE_POOL = 25

MIN_IMAGE_COMPETITION_HOURS = 1
MAX_IMAGE_COMPETITION_HOURS = 2
TASK_TIME_DELAY = 15  # number of minutes we wait to retry an organic request
# how many times in total do we attempt to delay an organic request looking for miners
MAX_DELAY_TIMES = 6
# Maximum number of evaluation attempts when all scores are zero (including the first one)
MAX_EVAL_ATTEMPTS = 4
MODEL_SIZE_REQUIRING_2_GPUS = 35 * 10**9  # 35B params

# Tournament GPU requirement thresholds (in billions of parameters)
TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100 = 4.0
TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100 = 12.0
TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100 = 40.0

# Tournament task type GPU multipliers
TOURNAMENT_DPO_GPU_MULTIPLIER = 3
TOURNAMENT_GRPO_GPU_MULTIPLIER = 2
MODEL_SIZE_REQUIRING_3_GPUS = 75 * 10**9
MODEL_SIZE_REQUIRING_4_GPUS = 110 * 10**9

# scoring stuff  - NOTE: Will want to slowly make more exponential now we have auditing
TEST_SCORE_WEIGHTING = 0.7  # synth will be (1 - this)
SCORE_PENALTY = -1
FIRST_PLACE_SCORE = 3

SIGMOID_STEEPNESS = 9  # Higher = sharper transition
SIGMOID_SHIFT = 0.5  # Shifts sigmoid curve horizontally
SIGMOID_POWER = 0.75  # Higher = more extreme difference between high and low scores
LINEAR_WEIGHT = 0.05  # Weight for linear component (0-1) - benefits low scores
SIGMOID_WEIGHT = 0.7  # Weight for sigmoid component (0-1) - benefits high scores

REWEIGHTING_EXP = 1.0  # how much of a drop off from leader

SCORING_WINDOW = 7  # number of days over which we score
OUTLIER_STD_THRESHOLD = 2.0  # number of standard deviations from the mean to reject the outlier scores


# processing stuff
MAX_CONCURRENT_MINER_ASSIGNMENTS = 5
MAX_CONCURRENT_TASK_PREPS = 3
MAX_CONCURRENT_TRAININGS = 10
MAX_CONCURRENT_EVALUATIONS = 1
MAX_TIME_DELAY_TO_FIND_MINERS = 1  # hours

PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT = 0.35
PERCENTAGE_OF_INSTRUCT_TASKS_THAT_SHOULD_BE_CHAT = 0.5
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_IMAGE = 0.25
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO = 0.15
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO = (
    1
    - PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
    - PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_IMAGE
    - PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
)
PERCENTAGE_OF_IMAGE_SYNTHS_SHOULD_BE_STYLE = (
    0.5  # person synth chance is 1 minus this (only for sdxl models, flux is always person)
)
PROBABILITY_STYLE_COMBINATION = 0.5
PERSON_SYNTH_DS_PREFIX = "person"
PERSON_SYNTH_DOCKER_IMAGE = "diagonalge/person_synth:latest"
PERSON_SYNTH_CONTAINER_SAVE_PATH = "/app/avatars/"

# grpo synth
MIN_NUM_REWARD_FUNCTIONS = 1
MAX_NUM_REWARD_FUNCTIONS = 5
PERCENTAGE_REWARD_FUNCTIONS_GENERIC_FROM_LLM = 0.0
PERCENTAGE_REWARD_FUNCTIONS_GENERIC_FROM_DB = 1 - PERCENTAGE_REWARD_FUNCTIONS_GENERIC_FROM_LLM

# diffusion eval stuff
LORA_SDXL_WORKFLOW_PATH = "validator/evaluation/comfy_workflows/lora_sdxl.json"
LORA_SDXL_WORKFLOW_PATH_DIFFUSERS = "validator/evaluation/comfy_workflows/lora_sdxl_diffusers.json"
LORA_FLUX_WORKFLOW_PATH = "validator/evaluation/comfy_workflows/lora_flux.json"
CHECKPOINTS_SAVE_PATH = "validator/evaluation/ComfyUI/models/checkpoints"
UNET_SAVE_PATH = "validator/evaluation/ComfyUI/models/unet"
DIFFUSERS_PATH = "validator/evaluation/ComfyUI/models/diffusers"
LORAS_SAVE_PATH = "validator/evaluation/ComfyUI/models/loras"
DIFFUSION_HF_DEFAULT_FOLDER = "checkpoint"
DIFFUSION_HF_DEFAULT_CKPT_NAME = "last.safetensors"
DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT = 0.25
EVAL_DEFAULTS = {"sdxl": {"steps": 20, "cfg": 8, "denoise": 0.9}, "flux": {"steps": 35, "cfg": 100, "denoise": 0.75}}


# Max jobs
MAX_CONCURRENT_JOBS = 60
MAX_CONCURRENT_SYNTHETIC_JOBS = 12
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
INSTRUCT_TEXT_TASK_SCORE_WEIGHT = 0.35
IMAGE_TASK_SCORE_WEIGHT = 0.2
DPO_TASK_SCORE_WEIGHT = 0.15
GRPO_TASK_SCORE_WEIGHT = 1 - INSTRUCT_TEXT_TASK_SCORE_WEIGHT - IMAGE_TASK_SCORE_WEIGHT - DPO_TASK_SCORE_WEIGHT

SEVEN_DAY_SCORE_WEIGHT = 0.4
THREE_DAY_SCORE_WEIGHT = 0.3
ONE_DAY_SCORE_WEIGHT = 0.3

TOURNAMENT_TEXT_WEIGHT = 0.65
TOURNAMENT_IMAGE_WEIGHT = 0.35
TOURNAMENT_INTERVAL_HOURS = 24
BURN_REDUCTION_RATE = 5.0
MAX_BURN_REDUCTION = 0.85
BASE_REGULAR_WEIGHT = 0.325
BASE_TOURNAMENT_WEIGHT = 0.375

# Emission distribution when performance diff occurs
LEGACY_PERFORM_DIFF_EMISSION_GAIN_PERCENT = 0.25

# HF models cache management
CACHE_TAU_DAYS = 10  # Time constant (τ) for exponential decay in days
CACHE_MAX_LOOKUP_DAYS = 30  # Maximum number of days to look back for usage data
MAX_CACHE_SIZE_BYTES = 500 * 1024**3 if NETUID == 241 else 1000 * 1024**3  # in bytes
CACHE_CLEANUP_INTERVAL = 8 * 60 * 60  # in seconds

# Docker evaluation
DOCKER_EVAL_HF_CACHE_DIR = "/root/.cache/huggingface"

# DPO evaluation
TRL_DPO_FIELD_PROMPT = "prompt"
TRL_DPO_FIELD_CHOSEN = "chosen"
TRL_DPO_FIELD_REJECTED = "rejected"

# Miner performance constants
MINER_PERFORMANCE_CACHE_TTL = 3600
MINER_PERFORMANCE_CACHE_KEY_PREFIX = "miner_performance:"
DEFAULT_RECENT_SUBMISSIONS_LIMIT = 20
CHAIN_WEIGHT_DIVISOR = 65535

# GRPO evaluation
TRL_GRPO_FIELD_PROMPT = GRPO_DEFAULT_FIELD_PROMPT


MIN_SYNTH_JOBS_REQUIRED_PER_DAY = 3

# Default, fixed Hyperparameters
BETA_DPO = 0.1
BETA_GRPO = 0.04

# GRPO evaluation
GRPO_INITIAL_BATCH_SIZE = 32
GRPO_DEFAULT_NUM_GENERATIONS = 2

STANDARD_INSTRUCT_COLUMN = "instruct"
STANDARD_INPUT_COLUMN = "input"
STANDARD_OUTPUT_COLUMN = "output"
STANDARD_SYSTEM_COLUMN = "system"
STANDARD_GRPO_PROMPT_COLUMN = "prompt"
STANDARD_DPO_PROMPT_COLUMN = "prompt"
STANDARD_DPO_CHOSEN_COLUMN = "chosen"
STANDARD_DPO_REJECTED_COLUMN = "rejected"
STANDARD_CHAT_MESSAGES_COLUMN = "conversations"
STANDARD_CHAT_TEMPLATE = "chatml"
STANDARD_CHAT_ROLE_USER = "human"
STANDARD_CHAT_ROLE_ASSISTANT = "gpt"
STANDARD_CHAT_ROLE_FIELD = "from"
STANDARD_CHAT_CONTENT_FIELD = "value"


# Trainer endpoints

PROXY_TRAINING_IMAGE_ENDPOINT = "/v1/trainer/start_training"
GET_GPU_AVAILABILITY_ENDPOINT = "/v1/trainer/get_gpu_availability"
TASK_DETAILS_ENDPOINT = "/v1/trainer/{task_id}"
GET_RECENT_TASKS_ENDPOINT = "/v1/trainer/get_recent_tasks"

# Tournament constants
DEFAULT_PARTICIPANT_REPO = "https://github.com/rayonlabs/G.O.D"
DEFAULT_PARTICIPANT_COMMIT = "8631451156e2915070f77e5547ca0d5ed3d0eb8a"
