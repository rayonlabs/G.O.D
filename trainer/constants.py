DEFAULT_IMAGE_DOCKERFILE_PATH = "dockerfiles/standalone-image-trainer.dockerfile"
DEFAULT_TEXT_DOCKERFILE_PATH = "dockerfiles/standalone-text-trainer.dockerfile"
TRAINER_CHECKPOINTS_PATH = "/tmp/trainer/checkpoints"
TEMP_REPO_PATH = "/tmp/trainer/repos/"
IMAGE_CONTAINER_SAVE_PATH = "/app/checkpoints/"
TEXT_CONTAINER_SAVE_PATH = "/workspace/axolotl/outputs/"
TASKS_FILE_PATH = "trainer/task_history.json"
VOLUME_NAMES = ["checkpoints", "cache"]
HF_UPLOAD_DOCKER_IMAGE = "diagonalge/hf-uploader:latest"
TRAINER_DOWNLOADER_DOCKER_IMAGE = "diagonalge/trainer-downloader:latest"
CACHE_CLEANER_DOCKER_IMAGE = "diagonalge/cache-cleaner:latest"
IMAGE_TASKS_HF_SUBFOLDER_PATH = "checkpoints"
DEFAULT_TRAINING_CONTAINER_MEM_LIMIT = "24g"
DEFAULT_TRAINING_CONTAINER_NANO_CPUS = 8
CACHE_PATH = "/cache"
STALE_TASK_GRACE_MINUTES = 30

#cache stuff
CHECKPOINTS_DIR = "/checkpoints"
CACHE_MODELS_DIR = "/cache/models"
CACHE_DATASETS_DIR = "/cache/datasets"
CACHE_CLEANUP_CUTOFF_HOURS = 12
