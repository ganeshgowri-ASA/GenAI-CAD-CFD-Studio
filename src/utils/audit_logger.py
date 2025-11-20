"""Comprehensive audit trail system for project actions.

This module provides functionality to log, retrieve, and export audit trails
for all project actions including CAD operations, simulations, configurations,
and exports.

Attributes:
    ACTION_TYPES: Available action types for logging.

Examples:
    >>> from src.utils.audit_logger import log_action, get_audit_logs
    >>> log_action('user123', 'CAD_CREATE', {'file': 'model.step'})
    >>> logs = get_audit_logs(user='user123')
"""

import json
import os
import csv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Action types
ACTION_TYPES = [
    'CAD_CREATE',
    'CAD_MODIFY',
    'CAD_DELETE',
    'SIMULATION_RUN',
    'SIMULATION_STOP',
    'CONFIG_CHANGE',
    'EXPORT',
    'BACKUP',
    'RESTORE',
    'PROJECT_CREATE',
    'PROJECT_DELETE',
    'USER_LOGIN',
    'USER_LOGOUT',
    'API_CALL'
]


class AuditLogger:
    """Audit logger for tracking project actions.

    Attributes:
        log_dir: Directory where audit logs are stored.
        log_file: Path to the audit log JSON file.
    """

    def __init__(self, log_dir: str = 'projects/audit_logs'):
        """Initialize audit logger.

        Args:
            log_dir: Directory to store audit logs. Default is 'projects/audit_logs'.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'audit_log.json'

        # Initialize log file if it doesn't exist
        if not self.log_file.exists():
            self._write_logs([])

    def _read_logs(self) -> List[Dict[str, Any]]:
        """Read all audit logs from file.

        Returns:
            List of audit log entries.
        """
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            return []
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read audit logs: {str(e)}")
            return []

    def _write_logs(self, logs: List[Dict[str, Any]]) -> None:
        """Write audit logs to file.

        Args:
            logs: List of audit log entries to write.
        """
        try:
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to write audit logs: {str(e)}")

    def log_action(
        self,
        user: str,
        action_type: str,
        details: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Log an action to the audit trail.

        Args:
            user: Username or user identifier.
            action_type: Type of action (must be in ACTION_TYPES).
            details: Dictionary containing action details (no sensitive data).
            timestamp: Action timestamp. If None, uses current UTC time.

        Returns:
            Dictionary containing the logged entry.

        Raises:
            ValueError: If action_type is not valid.

        Examples:
            >>> logger = AuditLogger()
            >>> entry = logger.log_action(
            ...     'user123',
            ...     'CAD_CREATE',
            ...     {'file': 'model.step', 'size': 1024}
            ... )
            >>> print(entry['action_type'])
            'CAD_CREATE'
        """
        if action_type not in ACTION_TYPES:
            raise ValueError(
                f"Invalid action_type '{action_type}'. Must be one of {ACTION_TYPES}"
            )

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Sanitize details to remove sensitive data
        sanitized_details = self._sanitize_details(details)

        entry = {
            'id': self._generate_id(),
            'timestamp': timestamp.isoformat(),
            'user': user,
            'action_type': action_type,
            'details': sanitized_details
        }

        # Read existing logs, append new entry, and write back
        logs = self._read_logs()
        logs.append(entry)
        self._write_logs(logs)

        logger.info(f"Logged action: {action_type} by {user}")
        return entry

    def get_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user: Optional[str] = None,
        action_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve audit logs with optional filtering.

        Args:
            start_date: Filter logs after this date (inclusive).
            end_date: Filter logs before this date (inclusive).
            user: Filter logs by user.
            action_type: Filter logs by action type.
            limit: Maximum number of logs to return.

        Returns:
            List of filtered audit log entries, sorted by timestamp (newest first).

        Examples:
            >>> logger = AuditLogger()
            >>> logs = logger.get_audit_logs(user='user123', action_type='CAD_CREATE')
            >>> print(len(logs))
            5
        """
        logs = self._read_logs()

        # Filter by date range
        if start_date:
            logs = [
                log for log in logs
                if datetime.fromisoformat(log['timestamp']) >= start_date
            ]

        if end_date:
            logs = [
                log for log in logs
                if datetime.fromisoformat(log['timestamp']) <= end_date
            ]

        # Filter by user
        if user:
            logs = [log for log in logs if log['user'] == user]

        # Filter by action type
        if action_type:
            logs = [log for log in logs if log['action_type'] == action_type]

        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x['timestamp'], reverse=True)

        # Apply limit
        if limit:
            logs = logs[:limit]

        return logs

    def export_audit_report(
        self,
        format: str = 'csv',
        output_dir: Optional[str] = None,
        **filter_kwargs
    ) -> str:
        """Export audit report in specified format.

        Args:
            format: Export format ('csv', 'json', or 'txt'). Default is 'csv'.
            output_dir: Directory to save report. If None, uses log_dir.
            **filter_kwargs: Additional keyword arguments passed to get_audit_logs.

        Returns:
            Path to the exported report file.

        Raises:
            ValueError: If format is not supported.

        Examples:
            >>> logger = AuditLogger()
            >>> report_path = logger.export_audit_report(format='csv', user='user123')
            >>> print(report_path)
            'projects/audit_logs/audit_report_20250119_120000.csv'
        """
        if format not in ['csv', 'json', 'txt']:
            raise ValueError(f"Unsupported format '{format}'. Use 'csv', 'json', or 'txt'.")

        if output_dir is None:
            output_dir = self.log_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get filtered logs
        logs = self.get_audit_logs(**filter_kwargs)

        # Generate filename with timestamp
        timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filename = f"audit_report_{timestamp_str}.{format}"
        output_path = output_dir / filename

        # Export based on format
        if format == 'csv':
            self._export_csv(logs, output_path)
        elif format == 'json':
            self._export_json(logs, output_path)
        elif format == 'txt':
            self._export_txt(logs, output_path)

        logger.info(f"Exported audit report to {output_path}")
        return str(output_path)

    def _export_csv(self, logs: List[Dict[str, Any]], output_path: Path) -> None:
        """Export logs to CSV format.

        Args:
            logs: List of audit log entries.
            output_path: Path to save CSV file.
        """
        with open(output_path, 'w', newline='') as f:
            if not logs:
                f.write("No audit logs found.\n")
                return

            fieldnames = ['id', 'timestamp', 'user', 'action_type', 'details']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for log in logs:
                # Convert details dict to string for CSV
                log_copy = log.copy()
                log_copy['details'] = json.dumps(log['details'])
                writer.writerow(log_copy)

    def _export_json(self, logs: List[Dict[str, Any]], output_path: Path) -> None:
        """Export logs to JSON format.

        Args:
            logs: List of audit log entries.
            output_path: Path to save JSON file.
        """
        with open(output_path, 'w') as f:
            json.dump(logs, f, indent=2)

    def _export_txt(self, logs: List[Dict[str, Any]], output_path: Path) -> None:
        """Export logs to text format.

        Args:
            logs: List of audit log entries.
            output_path: Path to save text file.
        """
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AUDIT REPORT\n")
            f.write("=" * 80 + "\n\n")

            if not logs:
                f.write("No audit logs found.\n")
                return

            for log in logs:
                f.write(f"ID: {log['id']}\n")
                f.write(f"Timestamp: {log['timestamp']}\n")
                f.write(f"User: {log['user']}\n")
                f.write(f"Action: {log['action_type']}\n")
                f.write(f"Details: {json.dumps(log['details'], indent=2)}\n")
                f.write("-" * 80 + "\n\n")

    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize details to remove sensitive data.

        Args:
            details: Dictionary containing action details.

        Returns:
            Sanitized details dictionary.
        """
        # List of keys that should not be logged
        sensitive_keys = [
            'password', 'token', 'api_key', 'secret', 'credential',
            'private_key', 'auth', 'authorization'
        ]

        sanitized = {}
        for key, value in details.items():
            # Check if key contains sensitive keywords
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = value

        return sanitized

    def _generate_id(self) -> str:
        """Generate unique ID for log entry.

        Returns:
            Unique identifier string.
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
        return f"LOG-{timestamp}"

    def clear_logs(self, confirm: bool = False) -> None:
        """Clear all audit logs.

        Args:
            confirm: Must be True to confirm deletion.

        Raises:
            ValueError: If confirm is not True.

        Examples:
            >>> logger = AuditLogger()
            >>> logger.clear_logs(confirm=True)
        """
        if not confirm:
            raise ValueError("Must confirm log deletion with confirm=True")

        self._write_logs([])
        logger.warning("All audit logs have been cleared")


# Convenience functions for global usage
_default_logger = None


def get_default_logger() -> AuditLogger:
    """Get or create the default audit logger instance.

    Returns:
        Default AuditLogger instance.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger()
    return _default_logger


def log_action(
    user: str,
    action_type: str,
    details: Dict[str, Any],
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """Log an action using the default logger.

    Args:
        user: Username or user identifier.
        action_type: Type of action.
        details: Dictionary containing action details.
        timestamp: Action timestamp.

    Returns:
        Dictionary containing the logged entry.

    Examples:
        >>> entry = log_action('user123', 'CAD_CREATE', {'file': 'model.step'})
    """
    logger = get_default_logger()
    return logger.log_action(user, action_type, details, timestamp)


def get_audit_logs(**kwargs) -> List[Dict[str, Any]]:
    """Get audit logs using the default logger.

    Args:
        **kwargs: Keyword arguments passed to AuditLogger.get_audit_logs.

    Returns:
        List of filtered audit log entries.

    Examples:
        >>> logs = get_audit_logs(user='user123', limit=10)
    """
    logger = get_default_logger()
    return logger.get_audit_logs(**kwargs)


def export_audit_report(format: str = 'csv', **kwargs) -> str:
    """Export audit report using the default logger.

    Args:
        format: Export format.
        **kwargs: Keyword arguments passed to AuditLogger.export_audit_report.

    Returns:
        Path to the exported report file.

    Examples:
        >>> path = export_audit_report(format='json', user='user123')
    """
    logger = get_default_logger()
    return logger.export_audit_report(format=format, **kwargs)
