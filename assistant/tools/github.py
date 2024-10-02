from langchain_core.tools import tool
import json
import subprocess
import os
import requests
from assistant.config.settings import GITHUB_PERSONAL_TOKEN, CLONE_BASE_DIR


@tool
def create_github_repo(repo_name: str, visibility: str):
    """
    Create a GitHub repository with specified visibility.

    Args:
        repo_name (str): The name of the repository to create.
        visibility (str): Visibility of the repository, either 'public' or 'private'.

    Returns:
        dict: Response from the GitHub API.
    """
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {GITHUB_PERSONAL_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {"name": repo_name, "private": visibility == "private"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

@tool
def clone_github_repo(owner: str, repo_name: str) -> str:
    """
    Clone a GitHub repository to a local directory.

    Args:
        owner (str): The owner (username or organization) of the GitHub repository.
        repo_name (str): The name of the GitHub repository to clone.

    Returns:
        str: A message indicating success or failure.
    """
    repo_url = f"https://github.com/{owner}/{repo_name}.git"

    local_dir = os.path.join(CLONE_BASE_DIR, repo_name)

    if os.path.exists(local_dir):
        return f"Error: The directory {local_dir} already exists. Please specify a different location."

    result = subprocess.run(
        ["git", "clone", repo_url, local_dir],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return f"Repository cloned successfully into {local_dir}: {result.stdout}"
