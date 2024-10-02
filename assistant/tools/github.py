from langchain_core.tools import tool
import json
import requests
from assistant.config.settings import GITHUB_PERSONAL_TOKEN


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
