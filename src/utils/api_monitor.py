"""
API Monitor and Usage Tracker

This module provides comprehensive tracking of API calls, credits used,
success/failure rates, and performance metrics for various AI services.

Tracks:
- Zoo.dev API calls and KCL generation
- Build123d local engine usage
- Anthropic Claude API (Vision and Text)
- Adam.new rendering API
- OpenFOAM simulation runs

Stores metrics in session state for real-time dashboard display.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json
from pathlib import Path


logger = logging.getLogger(__name__)


class APIService(Enum):
    """Enumeration of tracked API services"""
    ZOO_DEV = "Zoo.dev"
    BUILD123D = "Build123d"
    ANTHROPIC_VISION = "Anthropic Vision"
    ANTHROPIC_TEXT = "Anthropic Text"
    ADAM_RENDERING = "Adam.new"
    OPENFOAM = "OpenFOAM"
    GMSH = "Gmsh"
    FREECAD = "FreeCAD"


class CallStatus(Enum):
    """Status of API call"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    PAYMENT_REQUIRED = "payment_required"


@dataclass
class APICall:
    """Represents a single API call"""
    service: APIService
    status: CallStatus
    timestamp: datetime
    duration_ms: float
    tokens_used: int = 0
    credits_used: float = 0.0
    cost_usd: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'service': self.service.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'tokens_used': self.tokens_used,
            'credits_used': self.credits_used,
            'cost_usd': self.cost_usd,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class ServiceMetrics:
    """Aggregated metrics for a service"""
    service: APIService
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    rate_limited_calls: int = 0
    payment_required_calls: int = 0
    total_tokens: int = 0
    total_credits: float = 0.0
    total_cost_usd: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    last_call: Optional[datetime] = None
    first_call: Optional[datetime] = None

    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'service': self.service.value,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'timeout_calls': self.timeout_calls,
            'rate_limited_calls': self.rate_limited_calls,
            'payment_required_calls': self.payment_required_calls,
            'total_tokens': self.total_tokens,
            'total_credits': self.total_credits,
            'total_cost_usd': self.total_cost_usd,
            'avg_duration_ms': self.avg_duration_ms,
            'min_duration_ms': self.min_duration_ms if self.min_duration_ms != float('inf') else 0.0,
            'max_duration_ms': self.max_duration_ms,
            'success_rate': self.success_rate(),
            'last_call': self.last_call.isoformat() if self.last_call else None,
            'first_call': self.first_call.isoformat() if self.first_call else None
        }


class APIMonitor:
    """
    Monitors and tracks API usage across multiple services.

    Can be used as a singleton or context manager.
    Stores data in memory with optional persistence to file.
    """

    def __init__(
        self,
        enable_persistence: bool = True,
        persistence_path: Optional[Path] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize API Monitor.

        Args:
            enable_persistence: Whether to persist data to disk
            persistence_path: Path to persistence file
            session_id: Optional session identifier
        """
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path or Path('.api_monitor_data.json')
        self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')

        # Storage
        self.calls: List[APICall] = []
        self.metrics: Dict[APIService, ServiceMetrics] = {
            service: ServiceMetrics(service=service)
            for service in APIService
        }

        # Session data
        self.session_start = datetime.now()
        self.session_metadata: Dict[str, Any] = {}

        # Load persisted data if available
        if enable_persistence and self.persistence_path.exists():
            self._load_from_file()

        logger.info(f"APIMonitor initialized (session={self.session_id})")

    def record_call(
        self,
        service: APIService,
        status: CallStatus,
        duration_ms: float,
        tokens_used: int = 0,
        credits_used: float = 0.0,
        cost_usd: float = 0.0,
        error_message: Optional[str] = None,
        **metadata
    ) -> APICall:
        """
        Record an API call.

        Args:
            service: The API service called
            status: Status of the call
            duration_ms: Duration in milliseconds
            tokens_used: Number of tokens consumed
            credits_used: Credits/units consumed
            cost_usd: Cost in USD
            error_message: Error message if failed
            **metadata: Additional metadata

        Returns:
            The recorded APICall object
        """
        try:
            # Create call record
            call = APICall(
                service=service,
                status=status,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                tokens_used=tokens_used,
                credits_used=credits_used,
                cost_usd=cost_usd,
                error_message=error_message,
                metadata=metadata
            )

            # Store call
            self.calls.append(call)

            # Update metrics
            self._update_metrics(call)

            # Persist if enabled
            if self.enable_persistence:
                self._save_to_file()

            logger.debug(f"Recorded {service.value} call: {status.value} ({duration_ms:.2f}ms)")
            return call

        except Exception as e:
            logger.error(f"Error recording API call: {e}", exc_info=True)
            # Return a dummy call on error
            return APICall(
                service=service,
                status=CallStatus.FAILURE,
                timestamp=datetime.now(),
                duration_ms=0,
                error_message=f"Recording failed: {str(e)}"
            )

    def _update_metrics(self, call: APICall) -> None:
        """Update aggregated metrics for a service"""
        metrics = self.metrics[call.service]

        # Update counts
        metrics.total_calls += 1

        if call.status == CallStatus.SUCCESS:
            metrics.successful_calls += 1
        elif call.status == CallStatus.FAILURE:
            metrics.failed_calls += 1
        elif call.status == CallStatus.TIMEOUT:
            metrics.timeout_calls += 1
        elif call.status == CallStatus.RATE_LIMITED:
            metrics.rate_limited_calls += 1
        elif call.status == CallStatus.PAYMENT_REQUIRED:
            metrics.payment_required_calls += 1

        # Update usage
        metrics.total_tokens += call.tokens_used
        metrics.total_credits += call.credits_used
        metrics.total_cost_usd += call.cost_usd

        # Update timing
        if metrics.first_call is None:
            metrics.first_call = call.timestamp
        metrics.last_call = call.timestamp

        # Update duration stats
        metrics.min_duration_ms = min(metrics.min_duration_ms, call.duration_ms)
        metrics.max_duration_ms = max(metrics.max_duration_ms, call.duration_ms)

        # Recalculate average duration
        total_duration = metrics.avg_duration_ms * (metrics.total_calls - 1) + call.duration_ms
        metrics.avg_duration_ms = total_duration / metrics.total_calls

    def get_service_metrics(self, service: APIService) -> ServiceMetrics:
        """Get metrics for a specific service"""
        return self.metrics[service]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all services as dictionary"""
        return {
            service.value: metrics.to_dict()
            for service, metrics in self.metrics.items()
        }

    def get_total_cost(self) -> float:
        """Get total cost across all services"""
        return sum(m.total_cost_usd for m in self.metrics.values())

    def get_total_calls(self) -> int:
        """Get total number of calls across all services"""
        return sum(m.total_calls for m in self.metrics.values())

    def get_overall_success_rate(self) -> float:
        """Get overall success rate across all services"""
        total_calls = self.get_total_calls()
        if total_calls == 0:
            return 0.0

        total_successful = sum(m.successful_calls for m in self.metrics.values())
        return (total_successful / total_calls) * 100

    def get_recent_calls(
        self,
        service: Optional[APIService] = None,
        limit: int = 10,
        status_filter: Optional[CallStatus] = None
    ) -> List[APICall]:
        """
        Get recent API calls.

        Args:
            service: Filter by service (None for all)
            limit: Maximum number of calls to return
            status_filter: Filter by status (None for all)

        Returns:
            List of recent API calls
        """
        filtered_calls = self.calls

        if service:
            filtered_calls = [c for c in filtered_calls if c.service == service]

        if status_filter:
            filtered_calls = [c for c in filtered_calls if c.status == status_filter]

        # Sort by timestamp descending and limit
        filtered_calls = sorted(filtered_calls, key=lambda c: c.timestamp, reverse=True)
        return filtered_calls[:limit]

    def get_calls_in_timerange(
        self,
        start: datetime,
        end: datetime,
        service: Optional[APIService] = None
    ) -> List[APICall]:
        """Get calls within a time range"""
        filtered_calls = [
            c for c in self.calls
            if start <= c.timestamp <= end
        ]

        if service:
            filtered_calls = [c for c in filtered_calls if c.service == service]

        return sorted(filtered_calls, key=lambda c: c.timestamp)

    def get_hourly_stats(self, hours: int = 24) -> Dict[str, Dict[str, int]]:
        """
        Get hourly statistics for the last N hours.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dictionary mapping hour to service call counts
        """
        now = datetime.now()
        start_time = now - timedelta(hours=hours)

        hourly_data = defaultdict(lambda: defaultdict(int))

        for call in self.calls:
            if call.timestamp >= start_time:
                hour_key = call.timestamp.strftime('%Y-%m-%d %H:00')
                hourly_data[hour_key][call.service.value] += 1
                hourly_data[hour_key]['total'] += 1

        return dict(hourly_data)

    def clear_session_data(self) -> None:
        """Clear all session data"""
        logger.info("Clearing API monitor session data")
        self.calls.clear()
        self.metrics = {
            service: ServiceMetrics(service=service)
            for service in APIService
        }
        self.session_start = datetime.now()

        if self.enable_persistence:
            self._save_to_file()

    def export_to_dict(self) -> Dict[str, Any]:
        """Export all data to dictionary"""
        return {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'session_metadata': self.session_metadata,
            'metrics': self.get_all_metrics(),
            'total_calls': self.get_total_calls(),
            'total_cost': self.get_total_cost(),
            'overall_success_rate': self.get_overall_success_rate(),
            'calls': [call.to_dict() for call in self.calls]
        }

    def _save_to_file(self) -> None:
        """Save data to persistence file"""
        try:
            data = self.export_to_dict()
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved API monitor data to {self.persistence_path}")

        except Exception as e:
            logger.error(f"Failed to save API monitor data: {e}", exc_info=True)

    def _load_from_file(self) -> None:
        """Load data from persistence file"""
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)

            self.session_id = data.get('session_id', self.session_id)
            self.session_metadata = data.get('session_metadata', {})

            # Restore calls
            for call_data in data.get('calls', []):
                try:
                    call = APICall(
                        service=APIService(call_data['service']),
                        status=CallStatus(call_data['status']),
                        timestamp=datetime.fromisoformat(call_data['timestamp']),
                        duration_ms=call_data['duration_ms'],
                        tokens_used=call_data.get('tokens_used', 0),
                        credits_used=call_data.get('credits_used', 0.0),
                        cost_usd=call_data.get('cost_usd', 0.0),
                        error_message=call_data.get('error_message'),
                        metadata=call_data.get('metadata', {})
                    )
                    self.calls.append(call)
                    self._update_metrics(call)
                except Exception as e:
                    logger.warning(f"Failed to restore call: {e}")

            logger.info(f"Loaded {len(self.calls)} API calls from {self.persistence_path}")

        except Exception as e:
            logger.warning(f"Failed to load API monitor data: {e}")


# Global singleton instance
_global_monitor: Optional[APIMonitor] = None


def get_monitor(
    create_if_missing: bool = True,
    **kwargs
) -> Optional[APIMonitor]:
    """
    Get the global API monitor instance.

    Args:
        create_if_missing: Create instance if it doesn't exist
        **kwargs: Arguments for APIMonitor constructor

    Returns:
        Global APIMonitor instance or None
    """
    global _global_monitor

    if _global_monitor is None and create_if_missing:
        _global_monitor = APIMonitor(**kwargs)

    return _global_monitor


def reset_monitor() -> None:
    """Reset the global monitor instance"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.clear_session_data()


def record_api_call(
    service: APIService,
    status: CallStatus,
    duration_ms: float,
    **kwargs
) -> Optional[APICall]:
    """
    Convenience function to record an API call to the global monitor.

    Args:
        service: The API service
        status: Call status
        duration_ms: Duration in milliseconds
        **kwargs: Additional arguments

    Returns:
        APICall object or None if monitor not initialized
    """
    monitor = get_monitor()
    if monitor:
        return monitor.record_call(service, status, duration_ms, **kwargs)
    return None


# Context manager for timing API calls
class api_call_context:
    """
    Context manager for automatically timing and recording API calls.

    Usage:
        with api_call_context(APIService.ZOO_DEV) as ctx:
            result = make_api_call()
            ctx.set_tokens(result.tokens)
            ctx.set_cost(result.cost)
    """

    def __init__(
        self,
        service: APIService,
        auto_record: bool = True,
        monitor: Optional[APIMonitor] = None
    ):
        self.service = service
        self.auto_record = auto_record
        self.monitor = monitor or get_monitor()

        self.start_time = None
        self.end_time = None
        self.status = CallStatus.SUCCESS
        self.tokens_used = 0
        self.credits_used = 0.0
        self.cost_usd = 0.0
        self.error_message = None
        self.metadata = {}

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()

        # Determine status from exception
        if exc_type is not None:
            self.status = CallStatus.FAILURE
            self.error_message = str(exc_val)

        # Calculate duration
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        # Record if auto_record enabled
        if self.auto_record and self.monitor:
            self.monitor.record_call(
                service=self.service,
                status=self.status,
                duration_ms=duration_ms,
                tokens_used=self.tokens_used,
                credits_used=self.credits_used,
                cost_usd=self.cost_usd,
                error_message=self.error_message,
                **self.metadata
            )

        # Don't suppress exceptions
        return False

    def set_status(self, status: CallStatus) -> None:
        """Set the call status"""
        self.status = status

    def set_tokens(self, tokens: int) -> None:
        """Set tokens used"""
        self.tokens_used = tokens

    def set_credits(self, credits: float) -> None:
        """Set credits used"""
        self.credits_used = credits

    def set_cost(self, cost: float) -> None:
        """Set cost in USD"""
        self.cost_usd = cost

    def add_metadata(self, **kwargs) -> None:
        """Add metadata"""
        self.metadata.update(kwargs)
