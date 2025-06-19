from git import Repo
import os
from urllib.parse import urlparse

def clone_repo(repo_url: str, parent_dir: str, branch: str = None) -> str:
    repo_name = os.path.basename(urlparse(repo_url).path)
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    repo_dir = os.path.join(parent_dir, repo_name)

    if os.path.exists(repo_dir):
        print(f"Repository already exists at {repo_dir}. Skipping clone.")
        return repo_dir

    try:
        if branch:
            Repo.clone_from(repo_url, repo_dir, branch=branch)
        else:
            Repo.clone_from(repo_url, repo_dir)
        print(f"Repository cloned to {repo_dir}")
        return repo_dir
    except Exception as e:
        raise RuntimeError(f"Failed to clone repository: {e}")
