"""
API Usage Tracking System

Tracks API calls across different services (Zoo.dev, Claude, Adam.new, etc.)
with detailed metrics for dashboard display.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class APICallRecord:
    """Record of a single API call."""
    timestamp: str  # ISO format
    service: str  # 'zoo', 'claude', 'adam', 'build123d'
    model: Optional[str]  # Model name/version
    operation: str  # 'generate', 'analyze', 'render', etc.
    success: bool
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class APIUsageTracker:
    """
    Track API usage across all services.

    Features:
    - Per-service call counting
    - Success/failure tracking
    - Cost estimation
    - Token usage monitoring
    - Daily/weekly/monthly statistics
    - Persistent storage
    """

    # Cost per 1M tokens (in USD) - approximate values
    COST_PER_MILLION_TOKENS = {
        'claude-3-5-sonnet-20241022': {'input': 3.0, 'output': 15.0},
        'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0},
        'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
        'claude-3-sonnet-20240229': {'input': 3.0, 'output': 15.0},
    }

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize API usage tracker.

        Args:
            storage_path: Path to store usage data (defaults to ~/.genai_cad_cfd/api_usage.json)
        """
        if storage_path is None:
            storage_path = Path.home() / '.genai_cad_cfd' / 'api_usage.json'

        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._records: List[APICallRecord] = []
        self._lock = threading.Lock()

        # Load existing records
        self._load()

        logger.info(f"API Usage Tracker initialized. Storage: {self.storage_path}")

    def _load(self) -> None:
        """Load usage records from storage."""
        if not self.storage_path.exists():
            logger.info("No existing usage data found")
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self._records = [
                APICallRecord(**record)
                for record in data.get('records', [])
            ]

            logger.info(f"Loaded {len(self._records)} usage records")
        except Exception as e:
            logger.error(f"Failed to load usage data: {e}")
            self._records = []

    def _save(self) -> None:
        """Save usage records to storage."""
        try:
            data = {
                'records': [record.to_dict() for record in self._records],
                'last_updated': datetime.now().isoformat()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self._records)} usage records")
        except Exception as e:
            logger.error(f"Failed to save usage data: {e}")

    def record_call(
        self,
        service: str,
        operation: str,
        success: bool,
        model: Optional[str] = None,
        tokens_used: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        error_message: Optional[str] = None,
        **metadata
    ) -> None:
        """
        Record an API call.

        Args:
            service: Service name ('zoo', 'claude', 'adam', 'build123d')
            operation: Operation type ('generate', 'analyze', 'render', etc.)
            success: Whether the call succeeded
            model: Model name/version
            tokens_used: Number of tokens used
            duration_seconds: Call duration in seconds
            error_message: Error message if failed
            **metadata: Additional metadata
        """
        # Calculate cost if tokens provided
        cost_usd = None
        if tokens_used and model in self.COST_PER_MILLION_TOKENS:
            # Assume 75% input, 25% output for estimation
            input_tokens = int(tokens_used * 0.75)
            output_tokens = int(tokens_used * 0.25)

            costs = self.COST_PER_MILLION_TOKENS[model]
            cost_usd = (
                (input_tokens / 1_000_000) * costs['input'] +
                (output_tokens / 1_000_000) * costs['output']
            )

        record = APICallRecord(
            timestamp=datetime.now().isoformat(),
            service=service,
            model=model,
            operation=operation,
            success=success,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_seconds=duration_seconds,
            error_message=error_message,
            metadata=metadata if metadata else None
        )

        with self._lock:
            self._records.append(record)
            self._save()

        logger.info(f"Recorded API call: {service}/{operation} - {'SUCCESS' if success else 'FAILED'}")

    def get_today_stats(self) -> Dict[str, Any]:
        """Get statistics for today."""
        today = date.today().isoformat()

        with self._lock:
            today_records = [
                r for r in self._records
                if r.timestamp.startswith(today)
            ]

        return self._calculate_stats(today_records, "Today")

    def get_week_stats(self) -> Dict[str, Any]:
        """Get statistics for the past 7 days."""
        week_ago = (date.today() - timedelta(days=7)).isoformat()

        with self._lock:
            week_records = [
                r for r in self._records
                if r.timestamp >= week_ago
            ]

        return self._calculate_stats(week_records, "Past 7 Days")

    def get_month_stats(self) -> Dict[str, Any]:
        """Get statistics for the past 30 days."""
        month_ago = (date.today() - timedelta(days=30)).isoformat()

        with self._lock:
            month_records = [
                r for r in self._records
                if r.timestamp >= month_ago
            ]

        return self._calculate_stats(month_records, "Past 30 Days")

    def _calculate_stats(self, records: List[APICallRecord], period: str) -> Dict[str, Any]:
        """Calculate statistics from records."""
        if not records:
            return {
                'period': period,
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'success_rate': 0.0,
                'by_service': {},
                'by_model': {},
                'total_tokens': 0,
                'total_cost_usd': 0.0,
                'avg_duration_seconds': 0.0
            }

        # Overall statistics
        total_calls = len(records)
        successful_calls = sum(1 for r in records if r.success)
        failed_calls = total_calls - successful_calls
        success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0.0

        # By service
        by_service = defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0})
        for record in records:
            by_service[record.service]['total'] += 1
            if record.success:
                by_service[record.service]['success'] += 1
            else:
                by_service[record.service]['failed'] += 1

        # By model
        by_model = defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0, 'tokens': 0, 'cost_usd': 0.0})
        for record in records:
            if record.model:
                by_model[record.model]['total'] += 1
                if record.success:
                    by_model[record.model]['success'] += 1
                else:
                    by_model[record.model]['failed'] += 1
                if record.tokens_used:
                    by_model[record.model]['tokens'] += record.tokens_used
                if record.cost_usd:
                    by_model[record.model]['cost_usd'] += record.cost_usd

        # Aggregated metrics
        total_tokens = sum(r.tokens_used for r in records if r.tokens_used)
        total_cost_usd = sum(r.cost_usd for r in records if r.cost_usd)

        durations = [r.duration_seconds for r in records if r.duration_seconds]
        avg_duration_seconds = sum(durations) / len(durations) if durations else 0.0

        return {
            'period': period,
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': round(success_rate, 2),
            'by_service': dict(by_service),
            'by_model': dict(by_model),
            'total_tokens': total_tokens,
            'total_cost_usd': round(total_cost_usd, 4),
            'avg_duration_seconds': round(avg_duration_seconds, 2)
        }

    def get_service_breakdown(self, service: str, days: int = 7) -> Dict[str, Any]:
        """
        Get detailed breakdown for a specific service.

        Args:
            service: Service name ('zoo', 'claude', 'adam', 'build123d')
            days: Number of days to include

        Returns:
            Detailed statistics for the service
        """
        cutoff_date = (date.today() - timedelta(days=days)).isoformat()

        with self._lock:
            service_records = [
                r for r in self._records
                if r.service == service and r.timestamp >= cutoff_date
            ]

        if not service_records:
            return {
                'service': service,
                'period_days': days,
                'total_calls': 0,
                'operations': {}
            }

        # By operation
        operations = defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0})
        for record in service_records:
            operations[record.operation]['total'] += 1
            if record.success:
                operations[record.operation]['success'] += 1
            else:
                operations[record.operation]['failed'] += 1

        return {
            'service': service,
            'period_days': days,
            'total_calls': len(service_records),
            'successful_calls': sum(1 for r in service_records if r.success),
            'failed_calls': sum(1 for r in service_records if not r.success),
            'operations': dict(operations)
        }

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent error records.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of recent error records
        """
        with self._lock:
            error_records = [
                {
                    'timestamp': r.timestamp,
                    'service': r.service,
                    'model': r.model,
                    'operation': r.operation,
                    'error_message': r.error_message
                }
                for r in reversed(self._records)
                if not r.success and r.error_message
            ]

        return error_records[:limit]

    def clear_old_records(self, days: int = 90) -> int:
        """
        Clear records older than specified days.

        Args:
            days: Keep records from the last N days

        Returns:
            Number of records deleted
        """
        cutoff_date = (date.today() - timedelta(days=days)).isoformat()

        with self._lock:
            original_count = len(self._records)
            self._records = [
                r for r in self._records
                if r.timestamp >= cutoff_date
            ]
            deleted_count = original_count - len(self._records)

            if deleted_count > 0:
                self._save()
                logger.info(f"Cleared {deleted_count} old records (older than {days} days)")

        return deleted_count

    def export_to_csv(self, output_path: Path) -> None:
        """
        Export usage records to CSV.

        Args:
            output_path: Path to save CSV file
        """
        import csv

        with self._lock:
            records = self._records.copy()

        with open(output_path, 'w', newline='') as f:
            if not records:
                f.write("No records to export\n")
                return

            fieldnames = [
                'timestamp', 'service', 'model', 'operation', 'success',
                'tokens_used', 'cost_usd', 'duration_seconds', 'error_message'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in records:
                writer.writerow({
                    'timestamp': record.timestamp,
                    'service': record.service,
                    'model': record.model or '',
                    'operation': record.operation,
                    'success': record.success,
                    'tokens_used': record.tokens_used or '',
                    'cost_usd': record.cost_usd or '',
                    'duration_seconds': record.duration_seconds or '',
                    'error_message': record.error_message or ''
                })

        logger.info(f"Exported {len(records)} records to {output_path}")


# Global singleton instance
_tracker: Optional[APIUsageTracker] = None


def get_tracker() -> APIUsageTracker:
    """Get the global API usage tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = APIUsageTracker()
    return _tracker


def record_api_call(*args, **kwargs) -> None:
    """Convenience function to record an API call."""
    get_tracker().record_call(*args, **kwargs)
