import os
import shutil
from huggingface_hub import HfApi, login

def main():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    hf_user = os.getenv("HUGGINGFACE_USERNAME")
    task_id = os.getenv("TASK_ID")
    repo_name = os.getenv("EXPECTED_REPO_NAME")

    if not all([hf_token, hf_user, task_id, repo_name]):
        raise RuntimeError("Missing one or more required environment variables")

    login(token=hf_token)

    repo_id = f"{hf_user}/{repo_name}"
    local_folder = f"/app/checkpoints/{task_id}"

    if not os.path.isdir(local_folder):
        raise FileNotFoundError(f"Local folder {local_folder} does not exist")

    print(f"Creating repo {repo_id}...", flush=True)
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, private=False)

    print(f"Uploading contents of {local_folder} to {repo_id}...", flush=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_folder,
        commit_message=f"Upload task output {task_id}",
        token=hf_token
    )

    print(f"Uploaded successfully to https://huggingface.co/{repo_id}", flush=True)

if __name__ == "__main__":
    main()
