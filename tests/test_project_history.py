"""Comprehensive tests for Project History UI and utilities.

This module contains unit tests for:
- Version control integration (with mocked GitHub API)
- Backup/restore functionality
- Audit logging system
- UI components (with mocked Streamlit)

Target: >80% code coverage
"""

import pytest
import json
import os
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.audit_logger import (
    AuditLogger,
    log_action,
    get_audit_logs,
    export_audit_report,
    ACTION_TYPES
)
from src.utils.project_archiver import (
    ProjectArchiver,
    create_backup,
    list_backups,
    restore_backup,
    get_project_size
)


# ===========================
# Audit Logger Tests
# ===========================

class TestAuditLogger:
    """Test cases for AuditLogger class."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def logger(self, temp_log_dir):
        """Create AuditLogger instance with temp directory."""
        return AuditLogger(log_dir=temp_log_dir)

    def test_logger_initialization(self, logger, temp_log_dir):
        """Test logger initialization creates directory and file."""
        assert logger.log_dir == Path(temp_log_dir)
        assert logger.log_file.exists()

    def test_log_action_basic(self, logger):
        """Test logging a basic action."""
        entry = logger.log_action(
            user='test_user',
            action_type='CAD_CREATE',
            details={'file': 'test.step', 'size': 1024}
        )

        assert entry['user'] == 'test_user'
        assert entry['action_type'] == 'CAD_CREATE'
        assert entry['details']['file'] == 'test.step'
        assert 'id' in entry
        assert 'timestamp' in entry

    def test_log_action_invalid_type(self, logger):
        """Test logging with invalid action type raises ValueError."""
        with pytest.raises(ValueError):
            logger.log_action(
                user='test_user',
                action_type='INVALID_ACTION',
                details={}
            )

    def test_log_action_sanitizes_sensitive_data(self, logger):
        """Test that sensitive data is sanitized."""
        entry = logger.log_action(
            user='test_user',
            action_type='API_CALL',
            details={
                'endpoint': '/api/test',
                'api_key': 'secret123',
                'password': 'pass456'
            }
        )

        assert entry['details']['endpoint'] == '/api/test'
        assert entry['details']['api_key'] == '[REDACTED]'
        assert entry['details']['password'] == '[REDACTED]'

    def test_get_audit_logs_no_filters(self, logger):
        """Test getting all audit logs."""
        # Add multiple logs
        logger.log_action('user1', 'CAD_CREATE', {'file': 'model1.step'})
        logger.log_action('user2', 'SIMULATION_RUN', {'type': 'CFD'})
        logger.log_action('user1', 'CAD_MODIFY', {'file': 'model1.step'})

        logs = logger.get_audit_logs()
        assert len(logs) == 3

    def test_get_audit_logs_filter_by_user(self, logger):
        """Test filtering logs by user."""
        logger.log_action('user1', 'CAD_CREATE', {'file': 'model1.step'})
        logger.log_action('user2', 'SIMULATION_RUN', {'type': 'CFD'})
        logger.log_action('user1', 'CAD_MODIFY', {'file': 'model1.step'})

        logs = logger.get_audit_logs(user='user1')
        assert len(logs) == 2
        assert all(log['user'] == 'user1' for log in logs)

    def test_get_audit_logs_filter_by_action_type(self, logger):
        """Test filtering logs by action type."""
        logger.log_action('user1', 'CAD_CREATE', {'file': 'model1.step'})
        logger.log_action('user2', 'SIMULATION_RUN', {'type': 'CFD'})
        logger.log_action('user1', 'CAD_MODIFY', {'file': 'model1.step'})

        logs = logger.get_audit_logs(action_type='CAD_CREATE')
        assert len(logs) == 1
        assert logs[0]['action_type'] == 'CAD_CREATE'

    def test_get_audit_logs_filter_by_date_range(self, logger):
        """Test filtering logs by date range."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=2)
        future = now + timedelta(days=2)

        logger.log_action('user1', 'CAD_CREATE', {'file': 'model1.step'}, timestamp=past)
        logger.log_action('user2', 'SIMULATION_RUN', {'type': 'CFD'}, timestamp=now)
        logger.log_action('user3', 'CAD_MODIFY', {'file': 'model2.step'}, timestamp=future)

        logs = logger.get_audit_logs(
            start_date=now - timedelta(days=1),
            end_date=now + timedelta(days=1)
        )
        assert len(logs) == 1
        assert logs[0]['user'] == 'user2'

    def test_get_audit_logs_with_limit(self, logger):
        """Test limiting number of logs returned."""
        for i in range(10):
            logger.log_action(f'user{i}', 'CAD_CREATE', {'file': f'model{i}.step'})

        logs = logger.get_audit_logs(limit=5)
        assert len(logs) == 5

    def test_export_audit_report_csv(self, logger, temp_log_dir):
        """Test exporting audit report as CSV."""
        logger.log_action('user1', 'CAD_CREATE', {'file': 'model1.step'})
        logger.log_action('user2', 'SIMULATION_RUN', {'type': 'CFD'})

        report_path = logger.export_audit_report(format='csv', output_dir=temp_log_dir)

        assert Path(report_path).exists()
        assert report_path.endswith('.csv')

        with open(report_path, 'r') as f:
            content = f.read()
            assert 'id,timestamp,user,action_type,details' in content
            assert 'user1' in content
            assert 'CAD_CREATE' in content

    def test_export_audit_report_json(self, logger, temp_log_dir):
        """Test exporting audit report as JSON."""
        logger.log_action('user1', 'CAD_CREATE', {'file': 'model1.step'})

        report_path = logger.export_audit_report(format='json', output_dir=temp_log_dir)

        assert Path(report_path).exists()
        assert report_path.endswith('.json')

        with open(report_path, 'r') as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]['user'] == 'user1'

    def test_export_audit_report_txt(self, logger, temp_log_dir):
        """Test exporting audit report as text."""
        logger.log_action('user1', 'CAD_CREATE', {'file': 'model1.step'})

        report_path = logger.export_audit_report(format='txt', output_dir=temp_log_dir)

        assert Path(report_path).exists()
        assert report_path.endswith('.txt')

        with open(report_path, 'r') as f:
            content = f.read()
            assert 'AUDIT REPORT' in content
            assert 'user1' in content

    def test_export_audit_report_invalid_format(self, logger):
        """Test exporting with invalid format raises ValueError."""
        with pytest.raises(ValueError):
            logger.export_audit_report(format='xml')

    def test_clear_logs(self, logger):
        """Test clearing all logs."""
        logger.log_action('user1', 'CAD_CREATE', {'file': 'model1.step'})
        logger.log_action('user2', 'SIMULATION_RUN', {'type': 'CFD'})

        assert len(logger.get_audit_logs()) == 2

        logger.clear_logs(confirm=True)
        assert len(logger.get_audit_logs()) == 0

    def test_clear_logs_without_confirmation(self, logger):
        """Test clearing logs without confirmation raises ValueError."""
        with pytest.raises(ValueError):
            logger.clear_logs(confirm=False)

    def test_convenience_functions(self, temp_log_dir):
        """Test global convenience functions."""
        # Reset default logger
        import src.utils.audit_logger as audit_module
        audit_module._default_logger = AuditLogger(log_dir=temp_log_dir)

        entry = log_action('user1', 'CAD_CREATE', {'file': 'test.step'})
        assert entry['user'] == 'user1'

        logs = get_audit_logs(user='user1')
        assert len(logs) == 1

        report_path = export_audit_report(format='json', output_dir=temp_log_dir)
        assert Path(report_path).exists()


# ===========================
# Project Archiver Tests
# ===========================

class TestProjectArchiver:
    """Test cases for ProjectArchiver class."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory structure."""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir)

        # Create sample project structure
        (project_dir / 'projects' / 'models').mkdir(parents=True)
        (project_dir / 'projects' / 'results').mkdir(parents=True)
        (project_dir / 'projects' / 'configs').mkdir(parents=True)

        # Add sample files
        (project_dir / 'projects' / 'models' / 'model1.step').write_text('CAD model data')
        (project_dir / 'projects' / 'results' / 'result1.txt').write_text('Simulation results')
        (project_dir / 'projects' / 'configs' / 'config.json').write_text('{"key": "value"}')

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def archiver(self, temp_project_dir):
        """Create ProjectArchiver instance with temp directory."""
        backup_dir = Path(temp_project_dir) / 'backups'
        return ProjectArchiver(backup_dir=str(backup_dir), project_root=temp_project_dir)

    def test_archiver_initialization(self, archiver):
        """Test archiver initialization creates backup directory."""
        assert archiver.backup_dir.exists()
        assert archiver.project_root.exists()

    def test_create_backup_full(self, archiver):
        """Test creating a full backup."""
        backup_path = archiver.create_backup(
            project_name='test_project',
            include_models=True,
            include_results=True,
            include_configs=True,
            user='test_user',
            description='Test backup'
        )

        assert Path(backup_path).exists()
        assert backup_path.endswith('.zip')
        assert 'test_project' in backup_path

        # Verify backup contains metadata
        import zipfile
        with zipfile.ZipFile(backup_path, 'r') as zipf:
            assert 'backup_metadata.json' in zipf.namelist()
            metadata = json.loads(zipf.read('backup_metadata.json'))
            assert metadata['project_name'] == 'test_project'
            assert metadata['user'] == 'test_user'
            assert metadata['description'] == 'Test backup'

    def test_create_backup_selective(self, archiver):
        """Test creating a selective backup."""
        backup_path = archiver.create_backup(
            project_name='selective_backup',
            include_models=True,
            include_results=False,
            include_configs=False,
            user='test_user'
        )

        assert Path(backup_path).exists()

    def test_list_backups(self, archiver):
        """Test listing available backups."""
        # Create multiple backups
        archiver.create_backup('project1', user='user1')
        archiver.create_backup('project2', user='user2')

        backups = archiver.list_backups()

        assert len(backups) == 2
        assert all('filename' in b for b in backups)
        assert all('path' in b for b in backups)
        assert all('size' in b for b in backups)
        assert all('metadata' in b for b in backups)

    def test_list_backups_empty(self, archiver):
        """Test listing backups when none exist."""
        backups = archiver.list_backups()
        assert len(backups) == 0

    def test_restore_backup(self, archiver, temp_project_dir):
        """Test restoring from a backup."""
        # Create a backup
        backup_path = archiver.create_backup('restore_test', user='test_user')

        # Create a separate restore directory
        restore_dir = Path(temp_project_dir) / 'restore'
        restore_dir.mkdir()

        # Restore backup
        success = archiver.restore_backup(backup_path, target_dir=str(restore_dir))

        assert success is True
        assert (restore_dir / 'backup_metadata.json').exists()

    def test_restore_backup_nonexistent(self, archiver):
        """Test restoring from nonexistent backup raises error."""
        with pytest.raises(FileNotFoundError):
            archiver.restore_backup('nonexistent_backup.zip')

    def test_export_project_zip(self, archiver):
        """Test exporting project as ZIP."""
        export_path = archiver.export_project(
            project_name='export_test',
            format='zip'
        )

        assert Path(export_path).exists()
        assert export_path.endswith('.zip')

    def test_export_project_tar_gz(self, archiver):
        """Test exporting project as TAR.GZ."""
        export_path = archiver.export_project(
            project_name='export_test',
            format='tar.gz'
        )

        assert Path(export_path).exists()
        assert export_path.endswith('.tar.gz')

    def test_export_project_selective(self, archiver, temp_project_dir):
        """Test selective project export."""
        # Add some specific files
        (Path(temp_project_dir) / 'test.step').write_text('CAD data')
        (Path(temp_project_dir) / 'test.stl').write_text('Mesh data')
        (Path(temp_project_dir) / 'test.txt').write_text('Other data')

        export_path = archiver.export_project(
            project_name='selective_export',
            format='zip',
            selective=True,
            include_patterns=['*.step', '*.stl']
        )

        assert Path(export_path).exists()

    def test_export_project_invalid_format(self, archiver):
        """Test exporting with invalid format raises ValueError."""
        with pytest.raises(ValueError):
            archiver.export_project('test', format='rar')

    def test_get_project_size(self, archiver):
        """Test getting project size statistics."""
        size_info = archiver.get_project_size(include_backups=False)

        assert 'total_size' in size_info
        assert 'total_size_mb' in size_info
        assert 'total_size_gb' in size_info
        assert 'breakdown' in size_info
        assert size_info['total_size'] >= 0

    def test_get_project_size_with_backups(self, archiver):
        """Test getting project size including backups."""
        # Create a backup first
        archiver.create_backup('size_test', user='test_user')

        size_info = archiver.get_project_size(include_backups=True)

        assert 'backups' in size_info['breakdown']
        assert size_info['breakdown']['backups'] > 0

    def test_delete_backup(self, archiver):
        """Test deleting a backup."""
        # Create a backup
        backup_path = archiver.create_backup('delete_test', user='test_user')
        backup_filename = Path(backup_path).name

        # Verify it exists
        assert Path(backup_path).exists()

        # Delete it
        success = archiver.delete_backup(backup_filename)

        assert success is True
        assert not Path(backup_path).exists()

    def test_delete_backup_nonexistent(self, archiver):
        """Test deleting nonexistent backup returns False."""
        success = archiver.delete_backup('nonexistent_backup.zip')
        assert success is False

    def test_convenience_functions(self, temp_project_dir):
        """Test global convenience functions."""
        # Reset default archiver
        import src.utils.project_archiver as archiver_module
        backup_dir = Path(temp_project_dir) / 'backups'
        archiver_module._default_archiver = ProjectArchiver(
            backup_dir=str(backup_dir),
            project_root=temp_project_dir
        )

        # Create sample files (skip if already exists from fixture)
        models_dir = Path(temp_project_dir) / 'projects' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        backup_path = create_backup('test_project', user='test_user')
        assert Path(backup_path).exists()

        backups = list_backups()
        assert len(backups) > 0

        size_info = get_project_size()
        assert 'total_size' in size_info


# ===========================
# Version Control Tests (Mocked)
# ===========================

class TestVersionControl:
    """Test cases for version control integration with mocked GitHub API."""

    @pytest.fixture
    def mock_github(self):
        """Mock GitHub API client."""
        with patch('src.utils.version_control.Github') as mock:
            yield mock

    def test_check_github_availability(self):
        """Test checking GitHub availability."""
        from src.utils.version_control import check_github_availability
        # This will return True if PyGithub is installed
        result = check_github_availability()
        assert isinstance(result, bool)

    @patch('src.utils.version_control.GITHUB_AVAILABLE', True)
    @patch('src.utils.version_control.Github')
    def test_get_repository_info(self, mock_github_class):
        """Test getting repository information."""
        from src.utils.version_control import get_repository_info

        # Setup mock
        mock_client = MagicMock()
        mock_github_class.return_value = mock_client

        mock_repo = MagicMock()
        mock_repo.name = 'test-repo'
        mock_repo.full_name = 'owner/test-repo'
        mock_repo.description = 'Test repository'
        mock_repo.html_url = 'https://github.com/owner/test-repo'
        mock_repo.default_branch = 'main'
        mock_repo.created_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        mock_repo.updated_at = datetime(2023, 6, 1, tzinfo=timezone.utc)
        mock_repo.stargazers_count = 10
        mock_repo.forks_count = 5
        mock_repo.open_issues_count = 2
        mock_repo.language = 'Python'

        mock_client.get_repo.return_value = mock_repo

        # Test
        info = get_repository_info('owner/test-repo', 'fake_token')

        assert info['name'] == 'test-repo'
        assert info['full_name'] == 'owner/test-repo'
        assert info['stars'] == 10
        assert info['language'] == 'Python'

    @patch('src.utils.version_control.GITHUB_AVAILABLE', True)
    @patch('src.utils.version_control.Github')
    def test_list_branches(self, mock_github_class):
        """Test listing branches."""
        from src.utils.version_control import list_branches

        # Setup mock
        mock_client = MagicMock()
        mock_github_class.return_value = mock_client

        mock_repo = MagicMock()
        mock_branch1 = MagicMock()
        mock_branch1.name = 'main'
        mock_branch1.commit.sha = 'abc123'
        mock_branch1.protected = True

        mock_branch2 = MagicMock()
        mock_branch2.name = 'feature'
        mock_branch2.commit.sha = 'def456'
        mock_branch2.protected = False

        mock_repo.get_branches.return_value = [mock_branch1, mock_branch2]
        mock_repo.html_url = 'https://github.com/owner/test-repo'
        mock_client.get_repo.return_value = mock_repo

        # Test
        branches = list_branches('owner/test-repo', 'fake_token')

        assert len(branches) == 2
        assert branches[0]['name'] == 'main'
        assert branches[0]['protected'] is True
        assert branches[1]['name'] == 'feature'

    @patch('src.utils.version_control.GITHUB_AVAILABLE', True)
    @patch('src.utils.version_control.Github')
    def test_list_pull_requests(self, mock_github_class):
        """Test listing pull requests."""
        from src.utils.version_control import list_pull_requests

        # Setup mock
        mock_client = MagicMock()
        mock_github_class.return_value = mock_client

        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_pr.number = 123
        mock_pr.title = 'Test PR'
        mock_pr.state = 'open'
        mock_pr.merged = False
        mock_pr.user.login = 'test_user'
        mock_pr.created_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        mock_pr.updated_at = datetime(2023, 1, 2, tzinfo=timezone.utc)
        mock_pr.merged_at = None
        mock_pr.html_url = 'https://github.com/owner/test-repo/pull/123'
        mock_pr.head.ref = 'feature'
        mock_pr.base.ref = 'main'

        mock_repo.get_pulls.return_value = [mock_pr]
        mock_client.get_repo.return_value = mock_repo

        # Test
        prs = list_pull_requests('owner/test-repo', 'fake_token', state='open')

        assert len(prs) == 1
        assert prs[0]['number'] == 123
        assert prs[0]['title'] == 'Test PR'
        assert prs[0]['state'] == 'open'

    @patch('src.utils.version_control.GITHUB_AVAILABLE', True)
    @patch('src.utils.version_control.Github')
    def test_get_commit_history(self, mock_github_class):
        """Test getting commit history."""
        from src.utils.version_control import get_commit_history

        # Setup mock
        mock_client = MagicMock()
        mock_github_class.return_value = mock_client

        mock_repo = MagicMock()
        mock_repo.default_branch = 'main'

        mock_commit = MagicMock()
        mock_commit.sha = 'abc123'
        mock_commit.commit.message = 'Test commit'
        mock_commit.commit.author.name = 'Test Author'
        mock_commit.commit.author.email = 'test@example.com'
        mock_commit.commit.author.date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        mock_commit.html_url = 'https://github.com/owner/test-repo/commit/abc123'

        mock_repo.get_commits.return_value = [mock_commit]
        mock_client.get_repo.return_value = mock_repo

        # Test
        commits = get_commit_history('owner/test-repo', 'fake_token', limit=10)

        assert len(commits) == 1
        assert commits[0]['sha'] == 'abc123'
        assert commits[0]['message'] == 'Test commit'
        assert commits[0]['author'] == 'Test Author'

    @patch('src.utils.version_control.GITHUB_AVAILABLE', True)
    @patch('src.utils.version_control.Github')
    def test_compare_branches(self, mock_github_class):
        """Test comparing branches."""
        from src.utils.version_control import compare_branches

        # Setup mock
        mock_client = MagicMock()
        mock_github_class.return_value = mock_client

        mock_repo = MagicMock()
        mock_comparison = MagicMock()
        mock_comparison.ahead_by = 5
        mock_comparison.behind_by = 2
        mock_comparison.total_commits = 5
        mock_comparison.files = [MagicMock(additions=10, deletions=5)]
        mock_comparison.html_url = 'https://github.com/owner/test-repo/compare/main...feature'
        mock_comparison.commits = []

        mock_repo.compare.return_value = mock_comparison
        mock_client.get_repo.return_value = mock_repo

        # Test
        diff = compare_branches('owner/test-repo', 'main', 'feature', 'fake_token')

        assert diff['ahead_by'] == 5
        assert diff['behind_by'] == 2
        assert diff['files_changed'] == 1
        assert diff['additions'] == 10
        assert diff['deletions'] == 5


# ===========================
# UI Tests (Mocked Streamlit)
# ===========================

class TestProjectHistoryUI:
    """Test cases for UI components with mocked Streamlit."""

    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit session state."""
        with patch('src.ui.project_history.st') as mock_st:
            mock_st.session_state = {
                'github_token': None,
                'repo_name': None,
                'current_user': 'test_user',
                'selected_backup': None
            }
            yield mock_st

    def test_init_session_state(self, mock_streamlit):
        """Test session state initialization."""
        from src.ui.project_history import init_session_state

        init_session_state()

        assert 'github_token' in mock_streamlit.session_state
        assert 'repo_name' in mock_streamlit.session_state
        assert 'current_user' in mock_streamlit.session_state

    def test_get_github_token_from_session(self, mock_streamlit):
        """Test getting GitHub token from session state."""
        from src.ui.project_history import get_github_token

        mock_streamlit.session_state['github_token'] = 'test_token'
        mock_streamlit.secrets = {}

        token = get_github_token()
        assert token == 'test_token'

    def test_get_repo_name_from_session(self, mock_streamlit):
        """Test getting repo name from session state."""
        from src.ui.project_history import get_repo_name

        mock_streamlit.session_state['repo_name'] = 'owner/repo'
        mock_streamlit.secrets = {}

        repo = get_repo_name()
        assert repo == 'owner/repo'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src', '--cov-report=term-missing'])
