import os


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

BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

VALI_CONFIG_PATH = "validator/test_axolotl.yml"


# api stuff should move this out to be shared by both miner and vali code?
START_TRAINING_ENDPOINT = "/start_training/"
TASK_OFFER_ENDPOINT = "/task_offer/"
SUBMISSION_ENDPOINT = "/get_latest_model_submission/"


GET_ALL_DATASETS_ENDPOINT = "https://content.gradients.io/datasets?limit=1000"
GET_ALL_MODELS_ENDPOINT = "https://content.gradients.io/models"
GET_COLUMNS_FOR_DATASET_ENDPOINT = "https://content.gradients.io/dataset/{dataset}/columns/suggest"


GET_ALL_DATASETS_ID = "dataset_id"
GET_ALL_MODELS_ID = "model_id"


# task stuff
HOW_MANY_TASKS_MINIMAL_AT_THE_SAME_TIME = 20
NUMBER_OF_MINUTES_BETWEEN_SYNTH_TASK_CHECK = 15


# data stuff
TEST_SIZE = 0.1
TRAIN_TEST_SPLIT_PERCENTAGE = 0.1
GET_SYNTH_DATA = True
MAX_SYNTH_DATA_POINTS = 100
ADDITIONAL_SYNTH_DATA_PERCENTAGE = 1.0  # same size as training set

# synth stuff
SYNTH_GEN_BATCH_SIZE = 2
SYNTH_MODEL_TEMPERATURE = 0.4
CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"
GPU_SERVER = os.getenv("GPU_SERVER")

SYNTH_MODEL = "chat-llama-3-1-70b"
PROMPT_GEN_ENDPOINT = "https://api.nineteen.ai/v1/chat/completions"
GRADIENTS_ENDPOINT = "https://api.gradients.io/validator-signup"
PROMPT_PATH = "validator/prompts.yml"
NINETEEN_API_KEY = os.getenv("NINETEEN_API_KEY")

# Task Stuff
MINIMUM_MINER_POOL = 1
MIN_IDEAL_NUM_MINERS_IN_POOL = 3
MAX_IDEAL_NUM_MINERS_IN_POOL = 8
MIN_COMPETITION_HOURS = 1
MAX_COMPETITION_HOURS = 5

# scoring stuff
SOFTMAX_TEMPERATURE = 0.5
TEST_SCORE_WEIGHTING = 0.8  # synth will be (1 - this)
TARGET_SCORE_RATIO = 1
MIN_TASK_SCORE = -0.001 
MAX_TASK_SCORE = 1.6
TASK_SCORE_THRESHOLD = 0.8

# processing stuff
MAX_CONCURRENT_MINER_ASSIGNMENTS = 5
MAX_CONCURRENT_TASK_PREPS = 3
MAX_CONCURRENT_TRAININGS = 10
MAX_CONCURRENT_EVALUATIONS = 1
MAX_TIME_DELAY_TO_FIND_MINERS = 1  # hours
