import os
import subprocess
import re
import glob
import wandb
import shutil

from huggingface_hub import HfApi
from huggingface_hub import login

            
def sync_wandb_logs(cache_dir: str):
    sync_root = os.path.join(cache_dir, "wandb")
    run_dirs = glob.glob(os.path.join(sync_root, "offline-run-*"))

    if not run_dirs:
        print("No offline runs found.")
        return

    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir).split("-")[-1]
        print(f"Syncing run: {run_dir}")

        try:
            proc = subprocess.run(
                ["wandb", "sync", "--include-offline", run_dir],
                check=True,
                capture_output=True,
                text=True
            )
            output = proc.stdout + proc.stderr
            match = re.search(r"https://wandb\.ai/\S+", output)
            run_url = match.group(0) if match else None

            if run_url:
                print(f"Synced W&B Run: {run_url}")

            print(f"Synced Run: {run_id}")
            shutil.rmtree(run_dir)
            print(f"Deleted synced folder: {run_dir}")

        except Exception as e:
            print(f"Failed to sync {run_dir}: {e}")

def main():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    hf_user = os.getenv("HUGGINGFACE_USERNAME")
    wandb_token = os.getenv("WANDB_TOKEN")
    task_id = os.getenv("TASK_ID")
    repo_name = os.getenv("EXPECTED_REPO_NAME")
    local_folder = os.getenv("LOCAL_FOLDER")
    repo_subfolder = os.getenv("HF_REPO_SUBFOLDER", None)
    wandb_logs_path = os.getenv("WANDB_LOGS_PATH", None)

    if repo_subfolder:
        repo_subfolder = repo_subfolder.strip("/")

    if not all([hf_token, hf_user, task_id, repo_name]):
        raise RuntimeError("Missing one or more required environment variables")

    login(token=hf_token)

    repo_id = f"{hf_user}/{repo_name}"

    if not os.path.isdir(local_folder):
        raise FileNotFoundError(f"Local folder {local_folder} does not exist")

    print(f"Creating repo {repo_id}...", flush=True)
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, private=False)

    print(f"Uploading contents of {local_folder} to {repo_id}", flush=True)
    if repo_subfolder:
        print(f"Uploading into subfolder: {repo_subfolder}", flush=True)

    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_folder,
        path_in_repo=repo_subfolder if repo_subfolder else None,
        commit_message=f"Upload task output {task_id}",
        token=hf_token,
    )

    print(f"Uploaded successfully to https://huggingface.co/{repo_id}", flush=True)

    if wandb_token:
        try:
            wandb.login(key=wandb_token)
            sync_wandb_logs(cache_dir=wandb_logs_path)
        except Exception as e:
            print(f"Failed to sync W&B logs: {e}", flush=True)


if __name__ == "__main__":
    main()
