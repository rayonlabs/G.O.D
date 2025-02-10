# Tables
NODES_TABLE = "nodes"
NODES_HISTORY_TABLE = "nodes_history"
TASKS_TABLE = "tasks"
TEXT_TASKS_TABLE = "text_tasks"
IMAGE_TASKS_TABLE = "image_tasks"
TASK_NODES_TABLE = "task_nodes"
SUBMISSIONS_TABLE = "submissions"
OFFER_RESPONSES_TABLE = "offer_responses"
LATEST_SCORES_URL_TABLE = "latest_scores_url"
IMAGE_TEXT_PAIRS_TABLE = "image_text_pairs"

# Node Table Columns
NODE_ID = "node_id"
HOTKEY = "hotkey"
COLDKEY = "coldkey"
IP = "ip"
IP_TYPE = "ip_type"
PORT = "port"
SYMMETRIC_KEY = "symmetric_key"
SYMMETRIC_KEY_UUID = "symmetric_key_uuid"
NETUID = "netuid"
NETWORK = "network"
STAKE = "stake"
TRUST = "trust"
VTRUST = "vtrust"
INCENTIVE = "incentive"
LAST_UPDATED = "last_updated"
PROTOCOL = "protocol"
OUR_VALIDATOR = "our_validator"
CREATED_TIMESTAMP = "created_timestamp"
UPDATED_TIMESTAMP = "updated_timestamp"
ASSIGNED_MINERS = "assigned_miners"

# Task Table Columns
TASK_ID = "task_id"
ACCOUNT_ID = "account_id"
MODEL_ID = "model_id"
DS = "ds"
STATUS = "status"
HOURS_TO_COMPLETE = "hours_to_complete"
TEST_DATA = "test_data"
TRAINING_DATA = "training_data"
MINER_SCORES = "miner_scores"
CREATED_AT = "created_at"
NEXT_DELAY_AT = "next_delay_at"
UPDATED_AT = "updated_at"
STARTED_AT = "started_at"
COMPLETED_AT = "completed_at"
TERMINATION_AT = "termination_at"
IS_ORGANIC = "is_organic"
TIMES_DELAYED = "times_delayed"
ASSIGNED_MINERS = "assigned_miners"
TASK_TYPE = "task_type"
TRAINING_REPO_BACKUP = "training_repo_backup"
N_EVAL_ATTEMPTS = "n_eval_attempts"
RESULT_MODEL_NAME = "result_model_name"

# Text Tasks Table Columns
FIELD_SYSTEM = "field_system"
FIELD_INSTRUCTION = "field_instruction"
FIELD_INPUT = "field_input"
FIELD_OUTPUT = "field_output"
FORMAT = "format"
NO_INPUT_FORMAT = "no_input_format"
SYNTHETIC_DATA = "synthetic_data"
FILE_FORMAT = "file_format"

# Image Text Pairs Table Columns
IMAGE_TEXT_PAIRS = "image_text_pairs"
IMAGE_URL = "image_url"
TEXT_URL = "text_url"
ID = "id"


# Submissions Table Columns
SUBMISSION_ID = "submission_id"
REPO = "repo"
CREATED_ON = "created_on"

# Task Nodes Table Columns
TASK_NODE_QUALITY_SCORE = "quality_score"

EXPECTED_REPO_NAME = "expected_repo_name"

TEST_LOSS = "test_loss"
SYNTH_LOSS = "synth_loss"
SCORE_REASON = "score_reason"

# Offer Responses Table Columns
OFFER_RESPONSE = "offer_response"


# Common Column Names (shared between tables)
QUALITY_SCORE = "quality_score"  # Used in both submissions and task_nodes
