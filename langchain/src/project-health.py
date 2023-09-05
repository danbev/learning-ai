import requests

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

import os
import datetime

def get_repo_info(username, repo):
    # URL for GitHub API to get repository information
    repo_url = f"https://api.github.com/repos/{username}/{repo}"
    repo_resp = requests.get(repo_url).json()
    print(f'{repo_resp=}')

    stars = repo_resp.get('stargazers_count', 0)

    commits_url = f"https://api.github.com/repos/{username}/{repo}/commits"
    commits_resp = requests.get(commits_url).json()

    commits = len(commits_resp)
    return stars, commits, repo_url

def prompt_chatgpt(stars, commits):
    # Account for deprecation of LLM model
    current_date = datetime.datetime.now().date()
    target_date = datetime.date(2024, 6, 12)
    if current_date > target_date:
        llm_model = "gpt-3.5-turbo"
    else:
        llm_model = "gpt-3.5-turbo-0301"

    print(f'"Going to prompt chatgpt ({llm_model})"')
    chat = ChatOpenAI(temperature=0.0, model=llm_model)

    template_string = """Categorise the following project based on the \
            number of stars, and the number of commits. The more stars and \
            commits the better the project is likely to be. \
    stars: {stars}
    commits: {commits}
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)
    health_message = prompt_template.format_messages(
                    commits=commits,
                    stars=stars)
    health_response = chat(health_message)
    print(health_response)

if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())

    username = "danbev"
    repo = "learning-v8"
    stars, commits, repo_url = get_repo_info(username, repo)
    print(f'{repo_url} has {stars} stars and {commits} commits')
    prompt_chatgpt(stars, commits)



