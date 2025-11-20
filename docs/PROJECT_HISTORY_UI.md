# Project History & Version Control UI

## Overview

The Project History UI provides comprehensive version control integration, project audit trails, CAD model history tracking, simulation archives, and backup management for the GenAI-CAD-CFD-Studio platform.

## Features

### 📊 GitHub Dashboard
- **PR/Branch Status**: View all pull requests and branches with their status
- **Commit History**: Browse commit history with detailed information
- **Branch Comparison**: Compare branches to see differences
- **Direct Links**: Quick access to GitHub PRs and branches

### 📋 Audit Trail System
- **Action Logging**: Track all user actions (CAD create/modify, simulations, exports)
- **Search & Filter**: Find specific actions by user, type, or date range
- **Export Reports**: Generate audit reports in CSV, JSON, or TXT format
- **Privacy-Compliant**: Automatically redacts sensitive data (tokens, passwords)

### 💾 Backup & Export Management
- **One-Click Backup**: Create complete project backups with metadata
- **Selective Backup**: Choose what to include (models, results, configs)
- **Restore Functionality**: Restore projects from any backup point
- **Project Export**: Export projects in ZIP or TAR.GZ format
- **Disk Usage**: Monitor project size and storage breakdown

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure GitHub Integration (Optional)

Create `.streamlit/secrets.toml`:

```toml
[github]
token = "your_github_personal_access_token"
repo = "owner/repository_name"
```

#### How to Get a GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a descriptive name (e.g., "GenAI-CAD-CFD-Studio")
4. Select scopes:
   - `repo` (for private repositories) **or**
   - `public_repo` (for public repositories only)
5. Click "Generate token"
6. Copy the token and add it to `secrets.toml`

**⚠️ Security Note**: Never commit `secrets.toml` to version control!

## Usage

### Running the UI

```bash
streamlit run src/ui/project_history.py
```

The application will open in your default web browser at `http://localhost:8501`.

### GitHub Dashboard

1. Navigate to "📊 GitHub Dashboard" from the sidebar
2. If configured, you'll see:
   - Repository statistics (stars, forks, issues)
   - Pull Requests tab with filtering by state
   - Branches tab with protection status
   - Commits tab with history for any branch
3. Use the branch comparison tool to see differences

**Without GitHub Token**: The UI will prompt you to enter credentials manually or show setup instructions.

### Audit Trail

1. Navigate to "📋 Audit Trail" from the sidebar
2. Use filters to search logs:
   - **User**: Filter by specific user
   - **Action Type**: Filter by action (CAD_CREATE, SIMULATION_RUN, etc.)
   - **Date Range**: Filter by date range
3. View detailed log entries in the expandable section
4. Export audit reports:
   - Select format (CSV, JSON, TXT)
   - Click "Export Report"
   - Download the generated file

#### Action Types

- `CAD_CREATE` - CAD model created
- `CAD_MODIFY` - CAD model modified
- `CAD_DELETE` - CAD model deleted
- `SIMULATION_RUN` - Simulation started
- `SIMULATION_STOP` - Simulation stopped
- `CONFIG_CHANGE` - Configuration changed
- `EXPORT` - Project exported
- `BACKUP` - Backup created
- `RESTORE` - Backup restored
- `PROJECT_CREATE` - New project created
- `PROJECT_DELETE` - Project deleted
- `USER_LOGIN` - User logged in
- `USER_LOGOUT` - User logged out
- `API_CALL` - API called

### Backup Management

#### Creating a Backup

1. Navigate to "💾 Backup & Export" → "📦 Create Backup"
2. Enter project name
3. Select what to include:
   - ✅ CAD Models
   - ✅ Simulation Results
   - ✅ Configurations
4. Add optional description
5. Click "Create Backup"
6. Download the generated ZIP file

#### Restoring a Backup

1. Navigate to "💾 Backup & Export" → "📂 Restore Backup"
2. Browse available backups (sorted by date, newest first)
3. Click "Restore" on the desired backup
4. Confirm the restoration
5. Files will be restored to the project directory

**⚠️ Warning**: Restoration will overwrite existing files!

#### Exporting Projects

1. Navigate to "💾 Backup & Export" → "📤 Export Project"
2. Enter project name
3. Select format (ZIP or TAR.GZ)
4. For selective export:
   - Check "Selective Export"
   - Enter file patterns (e.g., `*.step`, `*.stl`)
5. Click "Export Project"
6. Download the generated archive

#### Checking Project Size

1. Navigate to "💾 Backup & Export" → "📊 Project Size"
2. Optionally include backups in calculation
3. Click "Calculate Size"
4. View total size and breakdown by directory

## Programmatic API

### Audit Logging

```python
from src.utils.audit_logger import log_action, get_audit_logs, export_audit_report

# Log an action
log_action(
    user='john_doe',
    action_type='CAD_CREATE',
    details={'file': 'solar_panel.step', 'size': 2048}
)

# Get filtered logs
logs = get_audit_logs(
    user='john_doe',
    action_type='CAD_CREATE',
    limit=10
)

# Export report
report_path = export_audit_report(
    format='csv',
    user='john_doe'
)
```

### Backup/Restore

```python
from src.utils.project_archiver import create_backup, list_backups, restore_backup

# Create a backup
backup_path = create_backup(
    project_name='solar_pv_project',
    include_models=True,
    include_results=True,
    user='john_doe',
    description='Before major redesign'
)

# List backups
backups = list_backups()
for backup in backups:
    print(f"{backup['filename']}: {backup['size']} bytes")

# Restore backup
success = restore_backup(backup_path)
```

### Version Control

```python
from src.utils.version_control import (
    get_repository_info,
    list_branches,
    list_pull_requests,
    get_commit_history,
    compare_branches
)

token = "your_github_token"
repo = "owner/repo"

# Get repository info
info = get_repository_info(repo, token)
print(f"Stars: {info['stars']}, Forks: {info['forks']}")

# List branches
branches = list_branches(repo, token)
for branch in branches:
    print(f"{branch['name']}: {branch['sha'][:7]}")

# List pull requests
prs = list_pull_requests(repo, token, state='open')
for pr in prs:
    print(f"PR #{pr['number']}: {pr['title']}")

# Get commit history
commits = get_commit_history(repo, token, branch='main', limit=10)
for commit in commits:
    print(f"{commit['sha'][:7]}: {commit['message']}")

# Compare branches
diff = compare_branches(repo, 'main', 'feature-branch', token)
print(f"Ahead: {diff['ahead_by']}, Behind: {diff['behind_by']}")
```

## Architecture

### Directory Structure

```
GenAI-CAD-CFD-Studio/
├── src/
│   ├── ui/
│   │   └── project_history.py      # Main UI application
│   └── utils/
│       ├── version_control.py      # GitHub API integration
│       ├── audit_logger.py         # Audit trail system
│       └── project_archiver.py     # Backup/restore utilities
├── tests/
│   └── test_project_history.py     # Comprehensive tests
├── projects/
│   ├── backups/                    # Backup storage
│   ├── models/                     # CAD models
│   ├── results/                    # Simulation results
│   ├── configs/                    # Configurations
│   └── audit_logs/                 # Audit log storage
├── .streamlit/
│   └── secrets.toml                # GitHub credentials (gitignored)
├── requirements.txt                # Python dependencies
└── docs/
    └── PROJECT_HISTORY_UI.md       # This file
```

### Data Storage

- **Audit Logs**: Stored in `projects/audit_logs/audit_log.json`
- **Backups**: Stored in `projects/backups/*.zip`
- **Exports**: Stored in `projects/backups/*_export_*.zip`

## Security

### Best Practices

1. **Never commit secrets**: `.streamlit/secrets.toml` is gitignored
2. **Token Permissions**: Use minimal required GitHub permissions
3. **Audit Log Privacy**: Sensitive data (passwords, tokens) is automatically redacted
4. **Backup Security**: Backups are stored locally, not exposed publicly

### Sensitive Data Handling

The audit logger automatically redacts fields containing:
- `password`
- `token`
- `api_key`
- `secret`
- `credential`
- `private_key`
- `auth`
- `authorization`

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/test_project_history.py -v

# Run with coverage
pytest tests/test_project_history.py -v --cov=src --cov-report=term-missing

# Run specific test
pytest tests/test_project_history.py::TestAuditLogger::test_log_action_basic -v
```

### Test Coverage

The test suite achieves >80% code coverage and includes:

- ✅ Audit logger functionality
- ✅ Backup/restore operations
- ✅ GitHub API integration (mocked)
- ✅ UI components (mocked Streamlit)
- ✅ Error handling and edge cases

## Troubleshooting

### GitHub Integration Issues

**Problem**: "GitHub authentication required" error

**Solution**:
1. Check that `secrets.toml` exists in `.streamlit/` directory
2. Verify token has correct permissions
3. Test token manually: `gh auth status`

**Problem**: "GitHub API rate limit exceeded"

**Solution**:
1. Wait for rate limit to reset (check `X-RateLimit-Reset` header)
2. Use authenticated requests (higher rate limit)
3. Implement caching for frequently accessed data

### Backup/Restore Issues

**Problem**: "Failed to create backup" error

**Solution**:
1. Check disk space: `df -h`
2. Verify write permissions on `projects/backups/` directory
3. Check that source directories exist

**Problem**: Backup file is too large

**Solution**:
1. Use selective backup (exclude large result files)
2. Clean up old results before backing up
3. Use compression (TAR.GZ is more efficient than ZIP)

### Audit Log Issues

**Problem**: Logs not appearing in UI

**Solution**:
1. Check `projects/audit_logs/audit_log.json` exists
2. Verify JSON format is valid: `python -m json.tool audit_log.json`
3. Check file permissions

## Generic Design Principles

The Project History UI is designed to be **domain-agnostic**:

- ❌ **NO** hardcoded references to specific CAD domains (Solar PV, chambers, etc.)
- ✅ **YES** generic terminology (models, simulations, projects)
- ✅ **YES** extensible action types for audit logging
- ✅ **YES** flexible backup/export for arbitrary file structures
- ✅ **YES** works with ANY GitHub repository

## Future Enhancements

Potential improvements for future versions:

- [ ] Cloud backup integration (AWS S3, Google Cloud Storage)
- [ ] Advanced search with regex patterns
- [ ] Audit log database backend (SQLite/PostgreSQL)
- [ ] Real-time collaboration features
- [ ] Automated backup scheduling
- [ ] Git integration beyond GitHub (GitLab, Bitbucket)
- [ ] Version diff visualization for CAD models
- [ ] Backup encryption for sensitive projects

## Support

For issues, questions, or contributions:

1. Check this documentation
2. Review test cases for usage examples
3. Open an issue on GitHub
4. Contribute improvements via pull requests

## License

See LICENSE file in the repository root.

---

**Last Updated**: 2025-01-19
**Version**: 1.0.0
**Maintainer**: GenAI-CAD-CFD-Studio Team
