# Tournament Miner Documentation 🏆

This guide covers everything you need to know about creating and running a training repository for G.O.D (Gradient-On-Demand) tournaments.

## Overview

The tournament system builds and runs your training code in Docker containers. Your code will compete against other miners by training models on provided datasets within time and resource constraints.

## Tournament Registration

The miner already includes the required endpoint at `/training_repo/{task_type}` that returns:

```json
{
  "github_repo": "https://github.com/yourname/training-repo",
  "commit_hash": "abc123..."
}
```

Where `task_type` can be:
- `"text"` - For text-based tournaments (Instruct, DPO, GRPO, Chat)
- `"image"` - For image-based tournaments (SDXL, Flux)

## Docker-Based Architecture

### Supported Base Images

**For Text Tasks (Instruct, DPO, GRPO, Chat):**
```dockerfile
FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1
```

**For Image Tasks (SDXL, Flux):**
```dockerfile
FROM diagonalge/kohya_latest:latest
```

### Required Repository Structure

```
your-training-repo/
├── dockerfiles/
│   ├── standalone-text-trainer.dockerfile
│   └── standalone-image-trainer.dockerfile
├── scripts/
│   ├── text_trainer.py
│   └── image_trainer.py
├── configs/
└── requirements.txt
```

**Important:** The dockerfile paths must be exactly:
- `dockerfiles/standalone-text-trainer.dockerfile`
- `dockerfiles/standalone-image-trainer.dockerfile`

## CLI Arguments

Your training scripts accept these standardised CLI arguments:

### Text Training Arguments
```bash
--task-id             # Unique task identifier
--model               # Base model to finetune
--dataset             # S3 dataset URL
--dataset-type        # JSON structure of dataset (columns, format)
--task-type           # "InstructTextTask", "DpoTask", or "GrpoTask"
--expected-repo-name  # Expected HuggingFace repository name for upload
--hours-to-complete   # Time limit in hours for the job to finish
```

### Image Training Arguments
```bash
--task-id             # Unique task identifier
--model               # Base model to finetune (e.g., stabilityai/stable-diffusion-xl-base-1.0)
--dataset-zip         # S3 URL to dataset zip file
--model-type          # "sdxl" or "flux"
--expected-repo-name  # Expected HuggingFace repository name for upload
--hours-to-complete   # Time limit in hours for the job to finish
```

## WandB Logging for Your Training Analysis

Include WandB logging so you can analyze your training runs after tournaments complete:

```python
def create_config(task_id, model, dataset, dataset_type, file_format, output_dir, expected_repo_name=None, log_wandb=True):
    if log_wandb:
        config["wandb_runid"] = f"{task_id}_{expected_repo_name}"
        config["wandb_name"] = f"{task_id}_{expected_repo_name}"
        config["wandb_mode"] = "offline"  # Logs saved locally
        os.makedirs(train_cst.WANDB_LOGS_DIR, exist_ok=True)

def patch_wandb_symlinks(base_dir: str):
    """Handle WandB symlinks by converting to real files."""
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)
            if os.path.islink(full_path):
                target_path = os.readlink(full_path)
                try:
                    os.unlink(full_path)
                    if os.path.exists(target_path):
                        shutil.copy(target_path, full_path)
                    else:
                        pathlib.Path(full_path).touch()
                except Exception as e:
                    print(f"Symlink patch failed: {e}")

# Call after training completes
patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)
```

## Dataset Handling

### Text Datasets
- Always provided as S3 URLs
- Common formats: JSON, CSV, Parquet
- Dataset type parameter describes the structure (columns, format)

### Image Datasets
- Provided as S3 URLs to zip files
- Should contain images and metadata (captions)
- Your script must handle extraction and preparation

## Output Structure Requirements

**Critical:** The output paths are standardised and MUST NOT be changed. The uploader expects models at these exact locations.

For your reference, all the paths used in training can be found at:

```trainer/constants.py```

And the functions to construct the paths can be found at:

```trainer/utils/training_paths.py```

Here are some most important paths:


### Model Output Path
```python
output_dir = f"/app/checkpoints/{task_id}/{expected_repo_name}"
```

### Input Model Cache Path
```python
# Models are pre-downloaded to this location by the downloader container
model_path = f"/cache/models/{model.replace('/', '--')}"
```

### Image Dataset Path
```python
# Models are pre-downloaded to this location by the downloader container
model_path = f"/cache/datasets/{task_id}.zip"
```

### Text Dataset Path
```python
# Models are pre-downloaded to this location by the downloader container
model_path = f"/cache/datasets/{task_id}_train_data.json"
```

## Example Entrypoint Script

Create a simple entrypoint that passes arguments to your training script:

```bash
#!/bin/bash
set -e

# For text training
python3 /workspace/scripts/text_trainer.py "$@"

# For image training
python3 /workspace/scripts/image_trainer.py "$@"
```

## Testing Your Setup

Test scripts are provided to validate your implementation locally:

```bash
# Text task examples
./examples/run_instruct_task.sh
./examples/run_dpo_task.sh
./examples/run_grpo_task.sh

# Image task examples
./examples/run_image_task.sh
```

## Tournament Structure

Tournaments run continuously with 4-7 day duration and 24-hour gaps between tournaments. There are separate tournaments for:
- **Text**: Instruct, DPO, GRPO tasks
- **Image**: SDXL and Flux diffusion tasks

### Group Stage
- Miners are organized into groups of 6-8 participants
- Each group competes on 3 tasks
- Top 1-3 performers from each group advance to knockout rounds

### Knockout Rounds
- Single elimination format
- Runs when field is reduced to less than 16 miners
- Head-to-head competition

### Boss Round
- Tournament winner must face defending champion
- Must win by at least 5% margin to claim title
- Defending champion retains title unless clearly outperformed

### GPU Requirements
- Determined by model size and task type
- Resource limits are enforced (memory, CPU)
- Plan for efficient resource usage

## Common Pitfalls to Avoid

1. **Don't change output paths** - The uploader expects exact locations
2. **Don't hardcode paths** - Use provided constants
3. **Don't ignore time limits** - Respect the hours-to-complete parameter
4. **Don't skip validation** - Test with various model sizes and datasets
5. **Don't upload/download in the training script** - Training container is run with no access to internet or host machine

## Reference Implementation

The G.O.D repository provides base training scripts that you can customize:
- `scripts/text_trainer.py` - Base implementation for text tasks
- `scripts/image_trainer.py` - Base implementation for image tasks

These scripts handle all required functionality including dataset preparation, training configuration, and model saving. You can modify and enhance these scripts to improve performance.