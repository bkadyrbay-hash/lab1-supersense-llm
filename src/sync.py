"""Pushing result files to GitHub from Colab or Kaggle, so that a reclaimed session loses no finished run."""

import os
import subprocess
import sys

REPO = "github.com/bkadyrbay-hash/lab1-supersense-llm.git"


def github_token() -> str | None:
    """Secret GH_TOKEN of Colab or Kaggle, looked up at each call so it can be added while a notebook runs."""
    try:
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("GH_TOKEN")
        if "google.colab" in sys.modules:
            from google.colab import userdata
            return userdata.get("GH_TOKEN")
    except Exception:  # both secret stores raise their own errors when the secret is missing or not shared
        return None
    return None


def push_results(message: str, branch: str, author: tuple[str, str]) -> None:
    """Commit everything under results/ to `branch` as `author` (name, email); silent no-op without a token."""
    token = github_token()
    if token is None:
        return
    url = f"https://{token}@{REPO}"
    identity = ["-c", f"user.name={author[0]}", "-c", f"user.email={author[1]}"]
    steps = {"add": ["add", "results"], "commit": [*identity, "commit", "-q", "-m", message],
             "pull": ["pull", "-q", "--rebase", url, branch], "push": ["push", "-q", url, f"HEAD:{branch}"]}
    for step, args in steps.items():
        # Output is captured so that the token never appears in the notebook.
        if subprocess.run(["git", *args], capture_output=True).returncode:
            print(f"Could not push the results (git {step} failed)")
            return
