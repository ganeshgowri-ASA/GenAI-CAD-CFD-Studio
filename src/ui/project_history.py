"""Project History & Version Control UI.

This module provides a comprehensive Streamlit-based interface for:
- GitHub PR/Branch Status Dashboard
- Project Version Timeline
- Audit Trail System
- Backup & Export Management
- Search & Filter Tools

Examples:
    Run this module as a Streamlit app:
    $ streamlit run src/ui/project_history.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
from typing import Dict, List, Optional, Any
import logging

# Import utility modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.version_control import (
    get_repository_info,
    list_branches,
    list_pull_requests,
    get_commit_history,
    compare_branches,
    check_github_availability,
    GitHubIntegrationError
)
from src.utils.audit_logger import (
    AuditLogger,
    get_audit_logs,
    log_action,
    export_audit_report,
    ACTION_TYPES
)
from src.utils.project_archiver import (
    ProjectArchiver,
    create_backup,
    list_backups,
    restore_backup,
    export_project,
    get_project_size
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'github_token' not in st.session_state:
        st.session_state.github_token = None
    if 'repo_name' not in st.session_state:
        st.session_state.repo_name = None
    if 'current_user' not in st.session_state:
        st.session_state.current_user = 'default_user'
    if 'selected_backup' not in st.session_state:
        st.session_state.selected_backup = None


def get_github_token() -> Optional[str]:
    """Get GitHub token from secrets or session state.

    Returns:
        GitHub Personal Access Token or None.
    """
    # Try to get from Streamlit secrets
    try:
        if 'github' in st.secrets and 'token' in st.secrets['github']:
            return st.secrets['github']['token']
    except Exception:
        pass

    # Fall back to session state
    return st.session_state.get('github_token')


def get_repo_name() -> Optional[str]:
    """Get repository name from secrets or session state.

    Returns:
        Repository name in format 'owner/repo' or None.
    """
    # Try to get from Streamlit secrets
    try:
        if 'github' in st.secrets and 'repo' in st.secrets['github']:
            return st.secrets['github']['repo']
    except Exception:
        pass

    # Fall back to session state
    return st.session_state.get('repo_name')


def render_github_config():
    """Render GitHub configuration section."""
    st.sidebar.header("⚙️ GitHub Configuration")

    if not check_github_availability():
        st.sidebar.error("❌ PyGithub not installed. GitHub features disabled.")
        st.sidebar.info("Install with: `pip install PyGithub>=2.1.1`")
        return False

    token = get_github_token()
    repo = get_repo_name()

    if not token:
        st.sidebar.warning("⚠️ No GitHub token configured")
        st.sidebar.info(
            "Add token to `.streamlit/secrets.toml`:\n"
            "```toml\n"
            "[github]\n"
            "token = \"your_github_token\"\n"
            "repo = \"owner/repo\"\n"
            "```"
        )

        # Allow manual input
        manual_token = st.sidebar.text_input("GitHub Token (optional)", type="password")
        manual_repo = st.sidebar.text_input("Repository (owner/repo)", value=repo or "")

        if manual_token:
            st.session_state.github_token = manual_token
        if manual_repo:
            st.session_state.repo_name = manual_repo

        return bool(manual_token and manual_repo)

    if not repo:
        st.sidebar.warning("⚠️ No repository configured")
        manual_repo = st.sidebar.text_input("Repository (owner/repo)")
        if manual_repo:
            st.session_state.repo_name = manual_repo
        return False

    st.sidebar.success(f"✅ Connected to {repo}")
    return True


def render_github_dashboard():
    """Render GitHub PR/Branch Status Dashboard."""
    st.header("📊 GitHub Dashboard")

    token = get_github_token()
    repo = get_repo_name()

    if not token or not repo:
        st.warning("GitHub integration not configured. See sidebar for setup instructions.")
        return

    try:
        # Repository Info
        with st.spinner("Loading repository information..."):
            repo_info = get_repository_info(repo, token)

        if 'error' in repo_info:
            st.error(f"❌ {repo_info['error']}")
            return

        # Display repo info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⭐ Stars", repo_info.get('stars', 0))
        with col2:
            st.metric("🍴 Forks", repo_info.get('forks', 0))
        with col3:
            st.metric("🐛 Open Issues", repo_info.get('open_issues', 0))
        with col4:
            st.metric("💻 Language", repo_info.get('language', 'Unknown'))

        st.markdown(f"**Description:** {repo_info.get('description', 'No description')}")
        st.markdown(f"**URL:** [{repo_info.get('url', '')}]({repo_info.get('url', '')})")

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📌 Pull Requests", "🌿 Branches", "📝 Commits"])

        with tab1:
            render_pull_requests(repo, token)

        with tab2:
            render_branches(repo, token)

        with tab3:
            render_commits(repo, token)

    except GitHubIntegrationError as e:
        st.error(f"❌ GitHub API Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"GitHub dashboard error: {str(e)}", exc_info=True)


def render_pull_requests(repo: str, token: str):
    """Render pull requests list."""
    st.subheader("Pull Requests")

    state_filter = st.selectbox("Filter by state", ["all", "open", "closed"], index=1)

    with st.spinner("Loading pull requests..."):
        prs = list_pull_requests(repo, token, state=state_filter)

    if not prs:
        st.info(f"No {state_filter} pull requests found.")
        return

    # Display as dataframe
    pr_data = []
    for pr in prs:
        pr_data.append({
            'PR': f"#{pr['number']}",
            'Title': pr['title'],
            'State': pr['state'],
            'Merged': '✅' if pr['merged'] else '❌',
            'Author': pr['author'],
            'Created': pr['created_at'][:10] if pr['created_at'] else 'N/A',
            'Branch': f"{pr['head_branch']} → {pr['base_branch']}",
            'URL': pr['url']
        })

    df = pd.DataFrame(pr_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Expandable details
    with st.expander("View PR Details"):
        selected_pr = st.selectbox(
            "Select PR",
            options=[f"#{pr['number']} - {pr['title']}" for pr in prs]
        )
        if selected_pr:
            pr_num = int(selected_pr.split('#')[1].split(' - ')[0])
            pr = next((p for p in prs if p['number'] == pr_num), None)
            if pr:
                st.json(pr)


def render_branches(repo: str, token: str):
    """Render branches list."""
    st.subheader("Branches")

    with st.spinner("Loading branches..."):
        branches = list_branches(repo, token)

    if not branches:
        st.info("No branches found.")
        return

    # Display as dataframe
    branch_data = []
    for branch in branches:
        branch_data.append({
            'Branch': branch['name'],
            'Protected': '🔒' if branch['protected'] else '🔓',
            'SHA': branch['sha'][:7],
            'URL': branch['url']
        })

    df = pd.DataFrame(branch_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Branch comparison tool
    st.subheader("Compare Branches")
    col1, col2 = st.columns(2)
    with col1:
        base_branch = st.selectbox("Base Branch", [b['name'] for b in branches], key='base')
    with col2:
        compare_branch = st.selectbox("Compare Branch", [b['name'] for b in branches], key='compare')

    if st.button("Compare"):
        if base_branch == compare_branch:
            st.warning("Please select different branches to compare.")
        else:
            with st.spinner("Comparing branches..."):
                try:
                    diff = compare_branches(repo, base_branch, compare_branch, token)
                    if 'error' in diff:
                        st.error(f"❌ {diff['error']}")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Ahead by", diff['ahead_by'])
                        with col2:
                            st.metric("Behind by", diff['behind_by'])
                        with col3:
                            st.metric("Files Changed", diff['files_changed'])

                        st.markdown(f"**Additions:** +{diff['additions']} | **Deletions:** -{diff['deletions']}")

                        if diff['commits']:
                            st.subheader("Commits in Comparison")
                            for commit in diff['commits']:
                                st.markdown(f"- `{commit['sha'][:7]}` {commit['message']}")
                except GitHubIntegrationError as e:
                    st.error(f"❌ {str(e)}")


def render_commits(repo: str, token: str):
    """Render commit history."""
    st.subheader("Commit History")

    # Get branches for selection
    with st.spinner("Loading branches..."):
        branches = list_branches(repo, token)

    if not branches:
        st.warning("No branches found.")
        return

    selected_branch = st.selectbox("Select Branch", [b['name'] for b in branches])
    commit_limit = st.slider("Number of commits", 10, 100, 50)

    with st.spinner(f"Loading commits from {selected_branch}..."):
        commits = get_commit_history(repo, token, branch=selected_branch, limit=commit_limit)

    if not commits:
        st.info("No commits found.")
        return

    # Display commits
    for commit in commits:
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{commit['message'].split(chr(10))[0]}**")
                st.caption(f"by {commit['author']} • {commit['date'][:10] if commit['date'] else 'N/A'}")
            with col2:
                st.code(commit['sha'][:7])
            st.markdown("---")


def render_audit_trail():
    """Render Audit Trail System."""
    st.header("📋 Audit Trail")

    # Create audit logger instance
    audit_logger = AuditLogger()

    # Filter controls
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_user = st.text_input("Filter by User", placeholder="Leave empty for all")

    with col2:
        filter_action = st.selectbox("Filter by Action Type", ["All"] + ACTION_TYPES)

    with col3:
        date_range = st.date_input("Date Range", value=[], max_value=datetime.now())

    # Get filtered logs
    filter_kwargs = {}
    if filter_user:
        filter_kwargs['user'] = filter_user
    if filter_action != "All":
        filter_kwargs['action_type'] = filter_action
    if len(date_range) == 2:
        filter_kwargs['start_date'] = datetime.combine(date_range[0], datetime.min.time(), tzinfo=timezone.utc)
        filter_kwargs['end_date'] = datetime.combine(date_range[1], datetime.max.time(), tzinfo=timezone.utc)

    logs = audit_logger.get_audit_logs(**filter_kwargs)

    # Display summary
    st.metric("Total Log Entries", len(logs))

    # Display logs
    if logs:
        # Convert to dataframe for better display
        log_data = []
        for log in logs:
            log_data.append({
                'ID': log['id'],
                'Timestamp': log['timestamp'][:19],
                'User': log['user'],
                'Action': log['action_type'],
                'Details': str(log['details'])[:50] + '...' if len(str(log['details'])) > 50 else str(log['details'])
            })

        df = pd.DataFrame(log_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Expandable details
        with st.expander("View Log Details"):
            selected_log_id = st.selectbox("Select Log Entry", [log['id'] for log in logs])
            if selected_log_id:
                log = next((l for l in logs if l['id'] == selected_log_id), None)
                if log:
                    st.json(log)

        # Export options
        st.subheader("Export Audit Report")
        col1, col2 = st.columns([2, 1])
        with col1:
            export_format = st.selectbox("Export Format", ["csv", "json", "txt"])
        with col2:
            if st.button("Export Report"):
                with st.spinner("Generating report..."):
                    try:
                        report_path = audit_logger.export_audit_report(
                            format=export_format,
                            **filter_kwargs
                        )
                        st.success(f"✅ Report exported to: {report_path}")

                        # Provide download button
                        with open(report_path, 'rb') as f:
                            st.download_button(
                                label="Download Report",
                                data=f,
                                file_name=Path(report_path).name,
                                mime='application/octet-stream'
                            )
                    except Exception as e:
                        st.error(f"❌ Export failed: {str(e)}")
    else:
        st.info("No audit logs found matching the filters.")

    # Add test log entry
    with st.expander("➕ Add Test Log Entry"):
        st.caption("For testing purposes only")
        col1, col2 = st.columns(2)
        with col1:
            test_action = st.selectbox("Action Type", ACTION_TYPES, key='test_action')
        with col2:
            test_user = st.text_input("User", value=st.session_state.current_user, key='test_user')

        test_details = st.text_area("Details (JSON)", value='{"file": "test.step", "size": 1024}')

        if st.button("Add Log Entry"):
            try:
                import json
                details = json.loads(test_details)
                audit_logger.log_action(test_user, test_action, details)
                st.success("✅ Log entry added!")
                st.rerun()
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON in details")
            except Exception as e:
                st.error(f"❌ Failed to add log: {str(e)}")


def render_backup_management():
    """Render Backup & Export Management."""
    st.header("💾 Backup & Export Management")

    archiver = ProjectArchiver()

    # Tabs for different operations
    tab1, tab2, tab3, tab4 = st.tabs(["📦 Create Backup", "📂 Restore Backup", "📤 Export Project", "📊 Project Size"])

    with tab1:
        render_create_backup(archiver)

    with tab2:
        render_restore_backup(archiver)

    with tab3:
        render_export_project(archiver)

    with tab4:
        render_project_size(archiver)


def render_create_backup(archiver: ProjectArchiver):
    """Render backup creation interface."""
    st.subheader("Create New Backup")

    project_name = st.text_input("Project Name", value="my_project")

    col1, col2, col3 = st.columns(3)
    with col1:
        include_models = st.checkbox("Include CAD Models", value=True)
    with col2:
        include_results = st.checkbox("Include Simulation Results", value=True)
    with col3:
        include_configs = st.checkbox("Include Configurations", value=True)

    description = st.text_area("Backup Description (optional)", placeholder="Enter backup notes...")

    if st.button("Create Backup", type="primary"):
        if not project_name:
            st.error("❌ Please enter a project name")
            return

        with st.spinner("Creating backup..."):
            try:
                backup_path = archiver.create_backup(
                    project_name=project_name,
                    include_models=include_models,
                    include_results=include_results,
                    include_configs=include_configs,
                    user=st.session_state.current_user,
                    description=description
                )

                # Log the action
                log_action(
                    st.session_state.current_user,
                    'BACKUP',
                    {'project': project_name, 'backup_path': backup_path}
                )

                st.success(f"✅ Backup created successfully!")
                st.info(f"📁 Backup saved to: `{backup_path}`")

                # Provide download button
                if Path(backup_path).exists():
                    with open(backup_path, 'rb') as f:
                        st.download_button(
                            label="Download Backup",
                            data=f,
                            file_name=Path(backup_path).name,
                            mime='application/zip'
                        )
            except Exception as e:
                st.error(f"❌ Backup failed: {str(e)}")
                logger.error(f"Backup creation error: {str(e)}", exc_info=True)


def render_restore_backup(archiver: ProjectArchiver):
    """Render backup restoration interface."""
    st.subheader("Restore from Backup")

    # List available backups
    backups = archiver.list_backups()

    if not backups:
        st.info("No backups found.")
        return

    # Display backups
    st.write(f"**Found {len(backups)} backup(s)**")

    for backup in backups:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**{backup['filename']}**")
                if backup.get('metadata'):
                    metadata = backup['metadata']
                    st.caption(
                        f"Project: {metadata.get('project_name', 'Unknown')} | "
                        f"User: {metadata.get('user', 'Unknown')} | "
                        f"Files: {metadata.get('files_count', 0)}"
                    )
                    if metadata.get('description'):
                        st.caption(f"📝 {metadata['description']}")
                st.caption(f"Created: {backup['created'][:19]} | Size: {backup['size'] / 1024:.2f} KB")

            with col2:
                if st.button("Restore", key=f"restore_{backup['filename']}"):
                    st.session_state.selected_backup = backup['path']

            with col3:
                if st.button("Delete", key=f"delete_{backup['filename']}"):
                    if archiver.delete_backup(backup['filename']):
                        st.success(f"✅ Deleted {backup['filename']}")
                        st.rerun()
                    else:
                        st.error("❌ Delete failed")

            st.markdown("---")

    # Restore confirmation
    if st.session_state.selected_backup:
        st.warning("⚠️ Restoring will overwrite existing files!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Restore", type="primary"):
                with st.spinner("Restoring backup..."):
                    try:
                        success = archiver.restore_backup(st.session_state.selected_backup)
                        if success:
                            # Log the action
                            log_action(
                                st.session_state.current_user,
                                'RESTORE',
                                {'backup_path': st.session_state.selected_backup}
                            )
                            st.success("✅ Backup restored successfully!")
                            st.session_state.selected_backup = None
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Restore failed: {str(e)}")
        with col2:
            if st.button("Cancel"):
                st.session_state.selected_backup = None
                st.rerun()


def render_export_project(archiver: ProjectArchiver):
    """Render project export interface."""
    st.subheader("Export Project")

    project_name = st.text_input("Project Name", value="my_project", key='export_project')

    col1, col2 = st.columns(2)
    with col1:
        export_format = st.selectbox("Export Format", ["zip", "tar.gz"])
    with col2:
        selective = st.checkbox("Selective Export")

    if selective:
        st.info("Selective export allows you to include only specific file patterns.")
        include_patterns = st.text_area(
            "Include Patterns (one per line)",
            value="*.step\n*.stl\n*.obj",
            help="Glob patterns for files to include"
        )
        patterns = [p.strip() for p in include_patterns.split('\n') if p.strip()]
    else:
        patterns = None

    if st.button("Export Project", type="primary"):
        if not project_name:
            st.error("❌ Please enter a project name")
            return

        with st.spinner("Exporting project..."):
            try:
                export_path = archiver.export_project(
                    project_name=project_name,
                    format=export_format,
                    selective=selective,
                    include_patterns=patterns if selective else None
                )

                # Log the action
                log_action(
                    st.session_state.current_user,
                    'EXPORT',
                    {'project': project_name, 'format': export_format, 'export_path': export_path}
                )

                st.success(f"✅ Project exported successfully!")
                st.info(f"📁 Export saved to: `{export_path}`")

                # Provide download button
                if Path(export_path).exists():
                    with open(export_path, 'rb') as f:
                        st.download_button(
                            label="Download Export",
                            data=f,
                            file_name=Path(export_path).name,
                            mime='application/octet-stream'
                        )
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
                logger.error(f"Export error: {str(e)}", exc_info=True)


def render_project_size(archiver: ProjectArchiver):
    """Render project size information."""
    st.subheader("Project Disk Usage")

    include_backups = st.checkbox("Include Backups in Calculation", value=False)

    if st.button("Calculate Size"):
        with st.spinner("Calculating project size..."):
            try:
                size_info = archiver.get_project_size(include_backups=include_backups)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Size (MB)", f"{size_info['total_size_mb']:.2f}")
                with col2:
                    st.metric("Total Size (GB)", f"{size_info['total_size_gb']:.3f}")
                with col3:
                    st.metric("Total Size (Bytes)", f"{size_info['total_size']:,}")

                # Breakdown by directory
                st.subheader("Size Breakdown")
                breakdown_data = []
                for dir_name, size in size_info['breakdown'].items():
                    breakdown_data.append({
                        'Directory': dir_name,
                        'Size (MB)': f"{size / (1024 * 1024):.2f}",
                        'Size (Bytes)': f"{size:,}"
                    })

                if breakdown_data:
                    df = pd.DataFrame(breakdown_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No project directories found.")

            except Exception as e:
                st.error(f"❌ Size calculation failed: {str(e)}")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Project History & Version Control",
        page_icon="📚",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    # Page header
    st.title("📚 Project History & Version Control")
    st.markdown("---")

    # Sidebar configuration
    github_configured = render_github_config()

    # User selection
    st.sidebar.header("👤 Current User")
    current_user = st.sidebar.text_input("Username", value=st.session_state.current_user)
    st.session_state.current_user = current_user

    # Main content - Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📊 GitHub Dashboard", "📋 Audit Trail", "💾 Backup & Export"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About Project History UI**\n\n"
        "This interface provides comprehensive project tracking including:\n"
        "- GitHub integration for PR/branch status\n"
        "- Complete audit trail of all actions\n"
        "- Backup and restore functionality\n"
        "- Project export tools\n\n"
        "All features work with ANY CAD/CFD project type."
    )

    # Render selected page
    if page == "📊 GitHub Dashboard":
        if github_configured:
            render_github_dashboard()
        else:
            st.info("⚠️ Please configure GitHub integration in the sidebar to use this feature.")
            st.markdown("""
            ### GitHub Integration Setup

            To enable GitHub features, add your credentials to `.streamlit/secrets.toml`:

            ```toml
            [github]
            token = "your_personal_access_token"
            repo = "owner/repository_name"
            ```

            **How to get a GitHub Personal Access Token:**
            1. Go to GitHub Settings → Developer settings → Personal access tokens
            2. Generate new token (classic)
            3. Select scopes: `repo` (for private repos) or `public_repo` (for public repos)
            4. Copy the token and add it to secrets.toml
            """)

    elif page == "📋 Audit Trail":
        render_audit_trail()

    elif page == "💾 Backup & Export":
        render_backup_management()


if __name__ == "__main__":
    main()
