import argparse
import os
import shutil
import subprocess
import time


def pull_latest_docker_images():
    os.system("docker pull weightswandering/tuning_vali:latest")


def should_update_local(local_commit: str, remote_commit: str) -> bool:
    return local_commit != remote_commit


def clean_hf_datasets_cache():
    try:
        hf_cache_path = os.path.expanduser("~/.cache/huggingface/datasets/")
        if os.path.exists(hf_cache_path):
            shutil.rmtree(hf_cache_path)
            print(f"Cleaned Huggingface datasets cache at {hf_cache_path}")
    except Exception as e:
        print(f"Error cleaning Huggingface datasets cache: {e}")


def run_auto_updater():
    while not os.path.exists(".vali.env"):
        time.sleep(10)

    pull_latest_docker_images()

    launch_command = "task validator"
    os.system(launch_command)
    time.sleep(60)

    while True:
        current_branch = subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
        local_commit = subprocess.getoutput("git rev-parse HEAD")
        os.system("git fetch")
        remote_commit = subprocess.getoutput(f"git rev-parse origin/{current_branch}")

        if should_update_local(local_commit, remote_commit):
            reset_cmd = f"git reset --hard {remote_commit}"
            process = subprocess.Popen(reset_cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            if not error:
                clean_hf_datasets_cache()
                pull_latest_docker_images()
                os.system("./utils/autoupdate_validator_steps.sh")
                time.sleep(20)

        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run auto updates for a validator")
    args = parser.parse_args()
    run_auto_updater()
