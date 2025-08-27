MAX_TRAINING_ATTEMPTS = 2

# Orchestrator cycle intervals (in seconds)
FETCH_TASKS_CYCLE_INTERVAL = 15 * 60  # 20 minutes
PROCESS_PENDING_TASKS_CYCLE_INTERVAL = 15 * 60  # 20 minutes
MONITOR_TRAINING_TASKS_CYCLE_INTERVAL = 15 * 60  # 20 minutes
MOVE_COMPLETED_TASKS_CYCLE_INTERVAL = 15 * 60  # 20 minutes
PERIODIC_GPU_AVAILABILITY_UPDATE_INTERVAL = 15 * 60  # 15 minutes

TOURNAMENT_PENDING_CYCLE_INTERVAL = 15 * 60
TOURNAMENT_ACTIVE_CYCLE_INTERVAL = 15 * 60
TOURNAMENT_PREP_TASK_CYCLE_INTERVAL = 15 * 60
TOURNAMENT_PENDING_ROUND_CYCLE_INTERVAL = 15 * 60


# Retry intervals (in seconds)
GPU_AVAILABILITY_CHECK_RETRY_INTERVAL = 15 * 60  # 15 minutes
TRAINING_START_RETRY_INTERVAL = 1 * 60  # 15 minutes

# Trainer requests
TRAINER_HTTP_TIMEOUT = 30.0  # seconds
EXPECTED_TRAINING_START_MESSAGE = "Started Training!"


# Tournament structure constants
MAX_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND = 14
EXPECTED_GROUP_SIZE = 8
MIN_GROUP_SIZE = 6


TOURNAMENT_PARTICIPANT_PING_BATCH_SIZE = 50

# Tournament task allocation
TEXT_TASKS_PER_GROUP = 1
IMAGE_TASKS_PER_GROUP = 1
TOURNAMENT_GROUP_TARGET_NUM_TASKS = 3

# Final round task counts
FINAL_ROUND_IMAGE_TASKS = 3
FINAL_ROUND_TEXT_TASKS = 3

# Knockout round task counts
KNOCKOUT_PAIR_TASKS = 1

# Model size constants (in billions)
BIG_MODEL_MIN_SIZE_B = 12.0
BIG_MODEL_MAX_SIZE_B = 71.0
DEFAULT_MODEL_MIN_SIZE_B = 1
DEFAULT_MODEL_MAX_SIZE_B = 10
MODEL_SIZE_RANGE_MULTIPLIER_MIN = 0.8
MODEL_SIZE_RANGE_MULTIPLIER_MAX = 1.2

# Model parameter conversion
MODEL_PARAMS_TO_BILLIONS = 1e9

# Progressive championship threshold constants
FIRST_DEFENSE_THRESHOLD = 0.10  # 10% advantage needed on first defense after becoming champion
SECOND_DEFENSE_THRESHOLD = 0.075  # 7.5% advantage needed on second defense
STEADY_STATE_THRESHOLD = 0.05  # 5% advantage needed from third defense onwards

# Obfuscation detection constants
OBFUSCATION_DETECTION_PATH = "./validator/obfuscation_detection/anti_obfuscation"

# GitHub repository constants
WINNER_REPO_GITHUB_ORG = "gradients-opensource"
WINNER_REPO_POSITION_PREFIX = "position"
