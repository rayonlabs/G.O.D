DEFAULT_IMAGE_DOCKERFILE_PATH = "dockerfiles/standalone-image-trainer.dockerfile"
DEFAULT_TEXT_DOCKERFILE_PATH = "dockerfiles/standalone-text-trainer.dockerfile"
TRAINER_CHECKPOINTS_PATH = "/tmp/trainer/checkpoints"
TEMP_REPO_PATH = "/tmp/trainer/repos/"
TASKS_FILE_PATH = "trainer/task_history.json"
VOLUME_NAMES = ["checkpoints", "cache"]
HF_UPLOAD_DOCKER_IMAGE = "diagonalge/hf-uploader:latest"
TRAINER_DOWNLOADER_DOCKER_IMAGE = "diagonalge/trainer-downloader:latest"
CACHE_CLEANER_DOCKER_IMAGE = "diagonalge/cache-cleaner:latest"
IMAGE_TASKS_HF_SUBFOLDER_PATH = "checkpoints"
DEFAULT_TRAINING_CONTAINER_MEM_LIMIT = "24g"
DEFAULT_TRAINING_CONTAINER_NANO_CPUS = 8
STALE_TASK_GRACE_MINUTES = 10
CACHE_CLEANUP_CUTOFF_HOURS = 12

#TRAINING PATHS 
CACHE_ROOT_PATH = "/cache"
HUGGINGFACE_CACHE_PATH = "/cache/hf_cache"
OUTPUT_CHECKPOINTS_PATH = "/app/checkpoints/"
CACHE_MODELS_DIR = "/cache/models"
CACHE_DATASETS_DIR = "/cache/datasets"
IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH = "/workspace/core/config"
IMAGE_CONTAINER_CONFIG_SAVE_PATH = "/dataset/configs"
IMAGE_CONTAINER_IMAGES_PATH = "/dataset/images"
OUTPUT_CHECKPOINTS_PATH = "/app/checkpoints/"
TEXT_CONTAINER_SAVE_PATH = "/workspace/axolotl/outputs/"
AXOLOTL_DIRECTORIES = {
    "data": "/workspace/axolotl/data",
    "prepared": "/workspace/axolotl/data_prepared",
    "configs": "/workspace/axolotl/configs",
    "outputs": "/workspace/axolotl/outputs",
    "input": "/workspace/input_data",
    "root": "/workspace/axolotl",
    "src": "/workspace/axolotl/src/"
}
