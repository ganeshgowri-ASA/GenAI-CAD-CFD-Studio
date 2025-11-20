"""Project backup and restore utilities.

This module provides functionality for creating backups, restoring from backups,
and exporting projects in various formats.

Attributes:
    BACKUP_DIR: Default directory for storing backups.

Examples:
    >>> from src.utils.project_archiver import create_backup, list_backups
    >>> backup_path = create_backup('my_project')
    >>> backups = list_backups()
"""

import os
import json
import shutil
import zipfile
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default backup directory
BACKUP_DIR = Path('projects/backups')


class ProjectArchiver:
    """Project archiver for backup and restore operations.

    Attributes:
        backup_dir: Directory where backups are stored.
        project_root: Root directory of the project.
    """

    def __init__(self, backup_dir: Optional[str] = None, project_root: str = '.'):
        """Initialize project archiver.

        Args:
            backup_dir: Directory to store backups. If None, uses default BACKUP_DIR.
            project_root: Root directory of the project. Default is current directory.
        """
        self.backup_dir = Path(backup_dir) if backup_dir else BACKUP_DIR
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(project_root)

    def create_backup(
        self,
        project_name: str,
        include_models: bool = True,
        include_results: bool = True,
        include_configs: bool = True,
        user: str = 'system',
        description: str = ''
    ) -> str:
        """Create a backup of the project.

        Args:
            project_name: Name of the project.
            include_models: Include CAD model files. Default is True.
            include_results: Include simulation results. Default is True.
            include_configs: Include configuration files. Default is True.
            user: User creating the backup. Default is 'system'.
            description: Optional description of the backup.

        Returns:
            Path to the created backup ZIP file.

        Raises:
            IOError: If backup creation fails.

        Examples:
            >>> archiver = ProjectArchiver()
            >>> backup_path = archiver.create_backup('my_project', user='user123')
            >>> print(backup_path)
            'projects/backups/my_project_20250119_120000.zip'
        """
        timestamp = datetime.now(timezone.utc)
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{project_name}_{timestamp_str}.zip"
        backup_path = self.backup_dir / backup_filename

        try:
            # Create metadata
            metadata = {
                'project_name': project_name,
                'timestamp': timestamp.isoformat(),
                'user': user,
                'description': description,
                'include_models': include_models,
                'include_results': include_results,
                'include_configs': include_configs,
                'files': []
            }

            # Create ZIP archive
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                files_added = 0

                # Add models
                if include_models:
                    files_added += self._add_directory_to_zip(
                        zipf, 'projects/models', 'models'
                    )

                # Add results
                if include_results:
                    files_added += self._add_directory_to_zip(
                        zipf, 'projects/results', 'results'
                    )

                # Add configs
                if include_configs:
                    files_added += self._add_directory_to_zip(
                        zipf, 'projects/configs', 'configs'
                    )
                    # Also include main config files
                    config_files = ['.streamlit/secrets.toml', 'config.json', 'settings.yaml']
                    for config_file in config_files:
                        config_path = self.project_root / config_file
                        if config_path.exists():
                            zipf.write(config_path, config_file)
                            files_added += 1

                # Update metadata with file count
                metadata['files_count'] = files_added
                metadata['backup_size'] = backup_path.stat().st_size if backup_path.exists() else 0

                # Add metadata to archive
                zipf.writestr('backup_metadata.json', json.dumps(metadata, indent=2))

            logger.info(f"Created backup: {backup_path} ({files_added} files)")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            # Clean up partial backup
            if backup_path.exists():
                backup_path.unlink()
            raise IOError(f"Failed to create backup: {str(e)}")

    def _add_directory_to_zip(
        self,
        zipf: zipfile.ZipFile,
        source_dir: str,
        archive_dir: str
    ) -> int:
        """Add directory contents to ZIP archive.

        Args:
            zipf: ZipFile object.
            source_dir: Source directory to add.
            archive_dir: Directory name in archive.

        Returns:
            Number of files added.
        """
        source_path = self.project_root / source_dir
        if not source_path.exists():
            return 0

        files_added = 0
        for root, dirs, files in os.walk(source_path):
            for file in files:
                file_path = Path(root) / file
                arcname = archive_dir / file_path.relative_to(source_path)
                zipf.write(file_path, arcname)
                files_added += 1

        return files_added

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with metadata.

        Returns:
            List of dictionaries containing backup information:
                - filename: Backup filename
                - path: Full path to backup
                - timestamp: Creation timestamp
                - size: File size in bytes
                - metadata: Backup metadata (if available)

        Examples:
            >>> archiver = ProjectArchiver()
            >>> backups = archiver.list_backups()
            >>> for backup in backups:
            ...     print(backup['filename'])
        """
        backups = []

        if not self.backup_dir.exists():
            return backups

        for backup_file in self.backup_dir.glob('*.zip'):
            backup_info = {
                'filename': backup_file.name,
                'path': str(backup_file),
                'size': backup_file.stat().st_size,
                'created': datetime.fromtimestamp(
                    backup_file.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            }

            # Try to read metadata from backup
            try:
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    if 'backup_metadata.json' in zipf.namelist():
                        metadata_content = zipf.read('backup_metadata.json')
                        backup_info['metadata'] = json.loads(metadata_content)
            except Exception as e:
                logger.warning(f"Failed to read metadata from {backup_file.name}: {str(e)}")
                backup_info['metadata'] = None

            backups.append(backup_info)

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created'], reverse=True)

        return backups

    def restore_backup(self, backup_path: str, target_dir: Optional[str] = None) -> bool:
        """Restore project from a backup.

        Args:
            backup_path: Path to the backup ZIP file.
            target_dir: Directory to restore to. If None, uses project root.

        Returns:
            True if restore was successful, False otherwise.

        Raises:
            FileNotFoundError: If backup file does not exist.
            IOError: If restore operation fails.

        Examples:
            >>> archiver = ProjectArchiver()
            >>> success = archiver.restore_backup('projects/backups/my_project_20250119_120000.zip')
            >>> print(success)
            True
        """
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        if target_dir is None:
            target_dir = self.project_root
        else:
            target_dir = Path(target_dir)

        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Read metadata
                metadata = None
                if 'backup_metadata.json' in zipf.namelist():
                    metadata_content = zipf.read('backup_metadata.json')
                    metadata = json.loads(metadata_content)
                    logger.info(f"Restoring backup: {metadata.get('project_name', 'Unknown')}")

                # Extract all files
                zipf.extractall(target_dir)

            logger.info(f"Successfully restored backup to {target_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore backup: {str(e)}")
            raise IOError(f"Failed to restore backup: {str(e)}")

    def export_project(
        self,
        project_name: str,
        format: str = 'zip',
        selective: bool = False,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> str:
        """Export project in specified format.

        Args:
            project_name: Name of the project.
            format: Export format ('zip' or 'tar.gz'). Default is 'zip'.
            selective: Enable selective export. Default is False.
            include_patterns: List of glob patterns to include (if selective=True).
            exclude_patterns: List of glob patterns to exclude.

        Returns:
            Path to the exported archive file.

        Raises:
            ValueError: If format is not supported.
            IOError: If export operation fails.

        Examples:
            >>> archiver = ProjectArchiver()
            >>> export_path = archiver.export_project(
            ...     'my_project',
            ...     format='zip',
            ...     selective=True,
            ...     include_patterns=['*.step', '*.stl']
            ... )
        """
        if format not in ['zip', 'tar.gz']:
            raise ValueError(f"Unsupported format '{format}'. Use 'zip' or 'tar.gz'.")

        timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        export_filename = f"{project_name}_export_{timestamp_str}.{format}"
        export_path = self.backup_dir / export_filename

        try:
            if format == 'zip':
                self._export_zip(
                    export_path, selective, include_patterns, exclude_patterns
                )
            else:  # tar.gz
                self._export_tar(
                    export_path, selective, include_patterns, exclude_patterns
                )

            logger.info(f"Exported project to {export_path}")
            return str(export_path)

        except Exception as e:
            logger.error(f"Failed to export project: {str(e)}")
            raise IOError(f"Failed to export project: {str(e)}")

    def _export_zip(
        self,
        output_path: Path,
        selective: bool,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> None:
        """Export project as ZIP archive.

        Args:
            output_path: Path to save ZIP file.
            selective: Enable selective export.
            include_patterns: Patterns to include.
            exclude_patterns: Patterns to exclude.
        """
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            self._add_files_to_archive(zipf, selective, include_patterns, exclude_patterns)

    def _export_tar(
        self,
        output_path: Path,
        selective: bool,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> None:
        """Export project as TAR.GZ archive.

        Args:
            output_path: Path to save TAR.GZ file.
            selective: Enable selective export.
            include_patterns: Patterns to include.
            exclude_patterns: Patterns to exclude.
        """
        with tarfile.open(output_path, 'w:gz') as tarf:
            self._add_files_to_archive(tarf, selective, include_patterns, exclude_patterns)

    def _add_files_to_archive(
        self,
        archive,
        selective: bool,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> None:
        """Add files to archive based on patterns.

        Args:
            archive: Archive object (ZipFile or TarFile).
            selective: Enable selective export.
            include_patterns: Patterns to include.
            exclude_patterns: Patterns to exclude.
        """
        if not selective:
            # Include everything except common exclusions
            exclude_patterns = exclude_patterns or [
                '**/__pycache__/**',
                '**/*.pyc',
                '**/.git/**',
                '**/venv/**',
                '**/.venv/**',
                '**/node_modules/**'
            ]

        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)

            # Filter directories
            if exclude_patterns:
                dirs[:] = [
                    d for d in dirs
                    if not any(root_path.joinpath(d).match(pattern) for pattern in exclude_patterns)
                ]

            for file in files:
                file_path = root_path / file

                # Check exclusions
                if exclude_patterns and any(file_path.match(pattern) for pattern in exclude_patterns):
                    continue

                # Check inclusions (if selective)
                if selective and include_patterns:
                    if not any(file_path.match(pattern) for pattern in include_patterns):
                        continue

                arcname = file_path.relative_to(self.project_root)

                if isinstance(archive, zipfile.ZipFile):
                    archive.write(file_path, arcname)
                else:  # tarfile
                    archive.add(file_path, arcname=arcname)

    def get_project_size(self, include_backups: bool = False) -> Dict[str, Any]:
        """Get disk usage statistics for the project.

        Args:
            include_backups: Include backup directory in calculation. Default is False.

        Returns:
            Dictionary containing size information:
                - total_size: Total size in bytes
                - total_size_mb: Total size in megabytes
                - total_size_gb: Total size in gigabytes
                - breakdown: Dictionary with sizes per directory

        Examples:
            >>> archiver = ProjectArchiver()
            >>> size_info = archiver.get_project_size()
            >>> print(f"Total: {size_info['total_size_mb']:.2f} MB")
        """
        total_size = 0
        breakdown = {}

        # Calculate size for main directories
        for dir_name in ['projects/models', 'projects/results', 'projects/configs']:
            dir_path = self.project_root / dir_name
            dir_size = self._get_directory_size(dir_path)
            breakdown[dir_name] = dir_size
            total_size += dir_size

        # Optionally include backups
        if include_backups:
            backup_size = self._get_directory_size(self.backup_dir)
            breakdown['backups'] = backup_size
            total_size += backup_size

        return {
            'total_size': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024),
            'breakdown': breakdown
        }

    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory.

        Args:
            directory: Path to directory.

        Returns:
            Total size in bytes.
        """
        if not directory.exists():
            return 0

        total_size = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                except OSError:
                    pass

        return total_size

    def delete_backup(self, backup_filename: str) -> bool:
        """Delete a backup file.

        Args:
            backup_filename: Name of the backup file to delete.

        Returns:
            True if deletion was successful, False otherwise.

        Examples:
            >>> archiver = ProjectArchiver()
            >>> success = archiver.delete_backup('my_project_20250119_120000.zip')
        """
        backup_path = self.backup_dir / backup_filename

        if not backup_path.exists():
            logger.warning(f"Backup file not found: {backup_filename}")
            return False

        try:
            backup_path.unlink()
            logger.info(f"Deleted backup: {backup_filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete backup: {str(e)}")
            return False


# Convenience functions for global usage
_default_archiver = None


def get_default_archiver() -> ProjectArchiver:
    """Get or create the default project archiver instance.

    Returns:
        Default ProjectArchiver instance.
    """
    global _default_archiver
    if _default_archiver is None:
        _default_archiver = ProjectArchiver()
    return _default_archiver


def create_backup(project_name: str, **kwargs) -> str:
    """Create a backup using the default archiver.

    Args:
        project_name: Name of the project.
        **kwargs: Additional keyword arguments passed to ProjectArchiver.create_backup.

    Returns:
        Path to the created backup file.

    Examples:
        >>> backup_path = create_backup('my_project', user='user123')
    """
    archiver = get_default_archiver()
    return archiver.create_backup(project_name, **kwargs)


def list_backups() -> List[Dict[str, Any]]:
    """List backups using the default archiver.

    Returns:
        List of backup information dictionaries.

    Examples:
        >>> backups = list_backups()
    """
    archiver = get_default_archiver()
    return archiver.list_backups()


def restore_backup(backup_path: str, **kwargs) -> bool:
    """Restore backup using the default archiver.

    Args:
        backup_path: Path to the backup file.
        **kwargs: Additional keyword arguments passed to ProjectArchiver.restore_backup.

    Returns:
        True if restore was successful.

    Examples:
        >>> success = restore_backup('projects/backups/my_project_20250119_120000.zip')
    """
    archiver = get_default_archiver()
    return archiver.restore_backup(backup_path, **kwargs)


def export_project(project_name: str, **kwargs) -> str:
    """Export project using the default archiver.

    Args:
        project_name: Name of the project.
        **kwargs: Additional keyword arguments passed to ProjectArchiver.export_project.

    Returns:
        Path to the exported file.

    Examples:
        >>> export_path = export_project('my_project', format='tar.gz')
    """
    archiver = get_default_archiver()
    return archiver.export_project(project_name, **kwargs)


def get_project_size(**kwargs) -> Dict[str, Any]:
    """Get project size using the default archiver.

    Args:
        **kwargs: Additional keyword arguments passed to ProjectArchiver.get_project_size.

    Returns:
        Dictionary with size information.

    Examples:
        >>> size_info = get_project_size(include_backups=True)
    """
    archiver = get_default_archiver()
    return archiver.get_project_size(**kwargs)
