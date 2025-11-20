"""GitHub API integration for version control operations.

This module provides functions to interact with GitHub API using PyGithub
for repository information, branches, pull requests, and commit history.

Attributes:
    None

Examples:
    >>> from src.utils.version_control import get_repository_info
    >>> repo_info = get_repository_info('owner/repo', 'github_token')
    >>> print(repo_info['name'])
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import os
import logging

try:
    from github import Github, GithubException, RateLimitExceededException
    from github.Repository import Repository
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    Github = None
    GithubException = Exception
    RateLimitExceededException = Exception
    Repository = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubIntegrationError(Exception):
    """Custom exception for GitHub integration errors."""
    pass


def _get_github_client(token: Optional[str] = None) -> Optional[Any]:
    """Get authenticated GitHub client.

    Args:
        token: GitHub Personal Access Token. If None, attempts to read from
            environment variable GITHUB_TOKEN.

    Returns:
        Authenticated Github client or None if authentication fails.

    Raises:
        GitHubIntegrationError: If PyGithub is not installed or authentication fails.
    """
    if not GITHUB_AVAILABLE:
        raise GitHubIntegrationError(
            "PyGithub is not installed. Install with: pip install PyGithub>=2.1.1"
        )

    if token is None:
        token = os.getenv('GITHUB_TOKEN')

    if not token:
        logger.warning("No GitHub token provided. Some features may be unavailable.")
        return None

    try:
        client = Github(token)
        # Test authentication
        client.get_user().login
        return client
    except GithubException as e:
        raise GitHubIntegrationError(f"GitHub authentication failed: {str(e)}")


def get_repository_info(repo_name: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Get repository metadata from GitHub.

    Args:
        repo_name: Repository name in format 'owner/repo'.
        token: GitHub Personal Access Token.

    Returns:
        Dictionary containing repository metadata:
            - name: Repository name
            - full_name: Full repository name (owner/repo)
            - description: Repository description
            - url: Repository URL
            - default_branch: Default branch name
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - stars: Number of stars
            - forks: Number of forks
            - open_issues: Number of open issues
            - language: Primary programming language

    Raises:
        GitHubIntegrationError: If repository access fails.

    Examples:
        >>> info = get_repository_info('owner/repo', 'github_token')
        >>> print(info['name'])
        'repo'
    """
    client = _get_github_client(token)
    if client is None:
        return {
            'error': 'GitHub authentication required',
            'name': repo_name.split('/')[-1] if '/' in repo_name else repo_name,
            'full_name': repo_name
        }

    try:
        repo = client.get_repo(repo_name)
        return {
            'name': repo.name,
            'full_name': repo.full_name,
            'description': repo.description or '',
            'url': repo.html_url,
            'default_branch': repo.default_branch,
            'created_at': repo.created_at.isoformat() if repo.created_at else None,
            'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
            'stars': repo.stargazers_count,
            'forks': repo.forks_count,
            'open_issues': repo.open_issues_count,
            'language': repo.language or 'Unknown'
        }
    except GithubException as e:
        logger.error(f"Failed to get repository info: {str(e)}")
        raise GitHubIntegrationError(f"Failed to get repository info: {str(e)}")


def list_branches(repo_name: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all branches in the repository with their status.

    Args:
        repo_name: Repository name in format 'owner/repo'.
        token: GitHub Personal Access Token.

    Returns:
        List of dictionaries containing branch information:
            - name: Branch name
            - sha: Commit SHA
            - protected: Whether branch is protected
            - url: Branch URL

    Raises:
        GitHubIntegrationError: If branch listing fails.

    Examples:
        >>> branches = list_branches('owner/repo', 'github_token')
        >>> for branch in branches:
        ...     print(branch['name'])
    """
    client = _get_github_client(token)
    if client is None:
        return []

    try:
        repo = client.get_repo(repo_name)
        branches = []

        for branch in repo.get_branches():
            branches.append({
                'name': branch.name,
                'sha': branch.commit.sha,
                'protected': branch.protected,
                'url': f"{repo.html_url}/tree/{branch.name}"
            })

        return branches
    except RateLimitExceededException:
        logger.error("GitHub API rate limit exceeded")
        raise GitHubIntegrationError("GitHub API rate limit exceeded. Please try again later.")
    except GithubException as e:
        logger.error(f"Failed to list branches: {str(e)}")
        raise GitHubIntegrationError(f"Failed to list branches: {str(e)}")


def list_pull_requests(
    repo_name: str,
    token: Optional[str] = None,
    state: str = 'all'
) -> List[Dict[str, Any]]:
    """List pull requests with their status.

    Args:
        repo_name: Repository name in format 'owner/repo'.
        token: GitHub Personal Access Token.
        state: PR state filter ('open', 'closed', 'all'). Default is 'all'.

    Returns:
        List of dictionaries containing PR information:
            - number: PR number
            - title: PR title
            - state: PR state (open/closed)
            - merged: Whether PR was merged
            - author: PR author username
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - merged_at: Merge timestamp (if merged)
            - url: PR URL
            - head_branch: Source branch
            - base_branch: Target branch

    Raises:
        GitHubIntegrationError: If PR listing fails.

    Examples:
        >>> prs = list_pull_requests('owner/repo', 'github_token', state='open')
        >>> for pr in prs:
        ...     print(f"PR #{pr['number']}: {pr['title']}")
    """
    client = _get_github_client(token)
    if client is None:
        return []

    try:
        repo = client.get_repo(repo_name)
        pulls = []

        for pr in repo.get_pulls(state=state, sort='updated', direction='desc'):
            pulls.append({
                'number': pr.number,
                'title': pr.title,
                'state': pr.state,
                'merged': pr.merged,
                'author': pr.user.login if pr.user else 'Unknown',
                'created_at': pr.created_at.isoformat() if pr.created_at else None,
                'updated_at': pr.updated_at.isoformat() if pr.updated_at else None,
                'merged_at': pr.merged_at.isoformat() if pr.merged_at else None,
                'url': pr.html_url,
                'head_branch': pr.head.ref if pr.head else 'Unknown',
                'base_branch': pr.base.ref if pr.base else 'Unknown'
            })

        return pulls
    except RateLimitExceededException:
        logger.error("GitHub API rate limit exceeded")
        raise GitHubIntegrationError("GitHub API rate limit exceeded. Please try again later.")
    except GithubException as e:
        logger.error(f"Failed to list pull requests: {str(e)}")
        raise GitHubIntegrationError(f"Failed to list pull requests: {str(e)}")


def get_commit_history(
    repo_name: str,
    token: Optional[str] = None,
    branch: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get commit history for a branch.

    Args:
        repo_name: Repository name in format 'owner/repo'.
        token: GitHub Personal Access Token.
        branch: Branch name. If None, uses default branch.
        limit: Maximum number of commits to retrieve. Default is 50.

    Returns:
        List of dictionaries containing commit information:
            - sha: Commit SHA
            - message: Commit message
            - author: Author name
            - author_email: Author email
            - date: Commit date
            - url: Commit URL

    Raises:
        GitHubIntegrationError: If commit retrieval fails.

    Examples:
        >>> commits = get_commit_history('owner/repo', 'github_token', branch='main', limit=10)
        >>> for commit in commits:
        ...     print(f"{commit['sha'][:7]}: {commit['message']}")
    """
    client = _get_github_client(token)
    if client is None:
        return []

    try:
        repo = client.get_repo(repo_name)

        # Use default branch if not specified
        if branch is None:
            branch = repo.default_branch

        commits = []
        for commit in repo.get_commits(sha=branch)[:limit]:
            commits.append({
                'sha': commit.sha,
                'message': commit.commit.message,
                'author': commit.commit.author.name if commit.commit.author else 'Unknown',
                'author_email': commit.commit.author.email if commit.commit.author else '',
                'date': commit.commit.author.date.isoformat() if commit.commit.author and commit.commit.author.date else None,
                'url': commit.html_url
            })

        return commits
    except RateLimitExceededException:
        logger.error("GitHub API rate limit exceeded")
        raise GitHubIntegrationError("GitHub API rate limit exceeded. Please try again later.")
    except GithubException as e:
        logger.error(f"Failed to get commit history: {str(e)}")
        raise GitHubIntegrationError(f"Failed to get commit history: {str(e)}")


def compare_branches(
    repo_name: str,
    base: str,
    compare: str,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """Compare two branches and return diff summary.

    Args:
        repo_name: Repository name in format 'owner/repo'.
        base: Base branch name.
        compare: Compare branch name.
        token: GitHub Personal Access Token.

    Returns:
        Dictionary containing comparison information:
            - ahead_by: Number of commits ahead
            - behind_by: Number of commits behind
            - total_commits: Total number of commits in comparison
            - files_changed: Number of files changed
            - additions: Number of additions
            - deletions: Number of deletions
            - url: Comparison URL
            - commits: List of commits in comparison

    Raises:
        GitHubIntegrationError: If branch comparison fails.

    Examples:
        >>> diff = compare_branches('owner/repo', 'main', 'feature-branch', 'github_token')
        >>> print(f"Ahead by {diff['ahead_by']} commits")
    """
    client = _get_github_client(token)
    if client is None:
        return {
            'error': 'GitHub authentication required',
            'ahead_by': 0,
            'behind_by': 0
        }

    try:
        repo = client.get_repo(repo_name)
        comparison = repo.compare(base, compare)

        commits = []
        for commit in comparison.commits:
            commits.append({
                'sha': commit.sha,
                'message': commit.commit.message,
                'author': commit.commit.author.name if commit.commit.author else 'Unknown',
                'date': commit.commit.author.date.isoformat() if commit.commit.author and commit.commit.author.date else None
            })

        return {
            'ahead_by': comparison.ahead_by,
            'behind_by': comparison.behind_by,
            'total_commits': comparison.total_commits,
            'files_changed': len(comparison.files),
            'additions': sum(f.additions for f in comparison.files),
            'deletions': sum(f.deletions for f in comparison.files),
            'url': comparison.html_url,
            'commits': commits
        }
    except RateLimitExceededException:
        logger.error("GitHub API rate limit exceeded")
        raise GitHubIntegrationError("GitHub API rate limit exceeded. Please try again later.")
    except GithubException as e:
        logger.error(f"Failed to compare branches: {str(e)}")
        raise GitHubIntegrationError(f"Failed to compare branches: {str(e)}")


def check_github_availability() -> bool:
    """Check if GitHub integration is available.

    Returns:
        True if PyGithub is installed, False otherwise.

    Examples:
        >>> if check_github_availability():
        ...     print("GitHub integration available")
    """
    return GITHUB_AVAILABLE
