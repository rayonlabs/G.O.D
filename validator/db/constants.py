# Tables
NODES_TABLE = "nodes"
NODES_HISTORY_TABLE = "nodes_history"
TASKS_TABLE = "tasks"
TASK_NODES_TABLE = "task_nodes"
SUBMISSIONS_TABLE = "submissions"

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

# Task Table Columns
TASK_ID = "task_id"
MODEL_ID = "model_id"
DS_ID = "ds_id"
SYSTEM = "system"
INSTRUCTION = "instruction"
INPUT = "input"
OUTPUT = "output"
FORMAT = "format"
NO_INPUT_FORMAT = "no_input_format"
STATUS = "status"
HOURS_TO_COMPLETE = "hours_to_complete"
USER_ID = "user_id"
TEST_DATA = "test_data"
SYNTHETIC_DATA = "synthetic_data"
TRAINING_DATA = "training_data"
MINER_SCORES = "miner_scores"
CREATED_TIMESTAMP = "created_timestamp"
DELAY_TIMESTAMP = "delay_timestamp"
UPDATED_TIMESTAMP = "updated_timestamp"
STARTED_TIMESTAMP = "started_timestamp"
COMPLETED_TIMESTAMP = "completed_timestamp"
END_TIMESTAMP = "end_timestamp"
IS_ORGANIC = "is_organic"
DELAY_TIMES = "delay_times"

# Submissions Table Columns
SUBMISSION_ID = "submission_id"
REPO = "repo"
CREATED_ON = "created_on"

# Task Nodes Table Columns
TASK_NODE_QUALITY_SCORE = "quality_score"

# Common Column Names (shared between tables)
QUALITY_SCORE = "quality_score"  # Used in both submissions and task_nodes
