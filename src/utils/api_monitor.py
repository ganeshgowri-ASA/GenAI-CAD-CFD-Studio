"""
API Monitoring and Metrics Tracking

This module provides comprehensive API call monitoring for:
- Zoo.dev API calls
- Build123d operations
- Anthropic Vision API calls
- Claude API calls

Features:
- Real-time metrics tracking
- Cost estimation
- Performance monitoring
- Session state integration
- Dashboard-ready data export

Author: GenAI CAD CFD Studio
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
import threading


logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """Supported API providers."""
    ZOO_DEV = "zoo_dev"
    BUILD123D = "build123d"
    ANTHROPIC_VISION = "anthropic_vision"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    OPENAI = "openai"


class APICallStatus(Enum):
    """Status of API call."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    PAYMENT_REQUIRED = "payment_required"
    RATE_LIMITED = "rate_limited"


@dataclass
class APICallMetrics:
    """Metrics for a single API call."""

    provider: APIProvider
    endpoint: str
    status: APICallStatus
    timestamp: datetime
    duration_ms: float
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'provider': self.provider.value,
            'endpoint': self.endpoint,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'request_size_bytes': self.request_size_bytes,
            'response_size_bytes': self.response_size_bytes,
            'tokens_used': self.tokens_used,
            'estimated_cost_usd': self.estimated_cost_usd,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class APIMonitor:
    """
    Comprehensive API monitoring and metrics tracking system.

    Features:
    - Track all API calls across providers
    - Calculate costs and usage statistics
    - Store metrics in memory and optionally to file
    - Thread-safe for concurrent operations
    - Session state integration
    - Dashboard-ready data export

    Example:
        >>> monitor = APIMonitor()
        >>> # Start tracking a call
        >>> call_id = monitor.start_call(APIProvider.ZOO_DEV, "/generate")
        >>> # ... make API call ...
        >>> monitor.end_call(call_id, APICallStatus.SUCCESS, tokens_used=1500)
        >>> # Get metrics
        >>> stats = monitor.get_summary_stats()
        >>> print(f"Total cost: ${stats['total_cost_usd']:.4f}")
    """

    # Pricing (USD per 1000 tokens) - Anthropic Claude pricing
    PRICING = {
        APIProvider.ANTHROPIC_CLAUDE: {
            'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075},
            'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015},
            'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
        },
        APIProvider.ANTHROPIC_VISION: {
            'vision': {'input': 0.003, 'output': 0.015}  # Same as Sonnet
        },
        APIProvider.ZOO_DEV: {
            'generation': {'per_request': 0.10}  # Estimated
        },
        APIProvider.BUILD123D: {
            'local': {'per_request': 0.0}  # Free local operation
        }
    }

    def __init__(
        self,
        enable_file_logging: bool = False,
        log_file_path: Optional[Path] = None,
        session_state: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize API Monitor.

        Args:
            enable_file_logging: Whether to log metrics to file
            log_file_path: Path to log file (default: ./logs/api_metrics.jsonl)
            session_state: Streamlit session state for persistence
        """
        self.metrics: List[APICallMetrics] = []
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        self.enable_file_logging = enable_file_logging
        self.log_file_path = log_file_path or Path('./logs/api_metrics.jsonl')
        self.session_state = session_state
        self._lock = threading.Lock()

        # Create log directory if needed
        if self.enable_file_logging:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize session state if provided
        if self.session_state is not None:
            if 'api_metrics' not in self.session_state:
                self.session_state['api_metrics'] = []

        logger.info(f"APIMonitor initialized (file_logging={enable_file_logging})")

    def start_call(
        self,
        provider: APIProvider,
        endpoint: str,
        request_size: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracking an API call.

        Args:
            provider: API provider
            endpoint: API endpoint or operation name
            request_size: Size of request in bytes
            metadata: Additional metadata

        Returns:
            Unique call ID for tracking
        """
        with self._lock:
            call_id = f"{provider.value}_{endpoint}_{datetime.now().timestamp()}"
            self.active_calls[call_id] = {
                'provider': provider,
                'endpoint': endpoint,
                'start_time': datetime.now(),
                'request_size': request_size,
                'metadata': metadata or {}
            }
            logger.debug(f"Started tracking API call: {call_id}")
            return call_id

    def end_call(
        self,
        call_id: str,
        status: APICallStatus,
        response_size: int = 0,
        tokens_used: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Optional[APICallMetrics]:
        """
        End tracking an API call and record metrics.

        Args:
            call_id: Call ID from start_call()
            status: Call status
            response_size: Size of response in bytes
            tokens_used: Total tokens used (if not split)
            input_tokens: Input tokens (for detailed tracking)
            output_tokens: Output tokens (for detailed tracking)
            model: Model name (for cost calculation)
            error_message: Error message if failed

        Returns:
            APICallMetrics object or None if call_id not found
        """
        with self._lock:
            if call_id not in self.active_calls:
                logger.warning(f"Call ID not found: {call_id}")
                return None

            call_data = self.active_calls.pop(call_id)
            end_time = datetime.now()
            duration_ms = (end_time - call_data['start_time']).total_seconds() * 1000

            # Calculate cost
            estimated_cost = self._calculate_cost(
                provider=call_data['provider'],
                tokens_used=tokens_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model
            )

            # Create metrics object
            metrics = APICallMetrics(
                provider=call_data['provider'],
                endpoint=call_data['endpoint'],
                status=status,
                timestamp=call_data['start_time'],
                duration_ms=duration_ms,
                request_size_bytes=call_data['request_size'],
                response_size_bytes=response_size,
                tokens_used=tokens_used or (input_tokens + output_tokens),
                estimated_cost_usd=estimated_cost,
                error_message=error_message,
                metadata={
                    **call_data['metadata'],
                    'model': model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                }
            )

            # Store metrics
            self.metrics.append(metrics)

            # Update session state
            if self.session_state is not None:
                self.session_state['api_metrics'].append(metrics.to_dict())

            # Log to file
            if self.enable_file_logging:
                self._log_to_file(metrics)

            logger.info(
                f"API call completed: {call_data['provider'].value}/{call_data['endpoint']} "
                f"- Status: {status.value}, Duration: {duration_ms:.0f}ms, "
                f"Tokens: {metrics.tokens_used}, Cost: ${estimated_cost:.4f}"
            )

            return metrics

    def record_call(
        self,
        provider: APIProvider,
        endpoint: str,
        status: APICallStatus,
        duration_ms: float,
        tokens_used: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> APICallMetrics:
        """
        Record a completed API call directly (without start/end tracking).

        Useful for wrapping existing API calls.

        Args:
            provider: API provider
            endpoint: API endpoint
            status: Call status
            duration_ms: Duration in milliseconds
            tokens_used: Total tokens used
            input_tokens: Input tokens
            output_tokens: Output tokens
            model: Model name
            error_message: Error message if failed
            metadata: Additional metadata

        Returns:
            APICallMetrics object
        """
        with self._lock:
            # Calculate cost
            estimated_cost = self._calculate_cost(
                provider=provider,
                tokens_used=tokens_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model
            )

            # Create metrics object
            metrics = APICallMetrics(
                provider=provider,
                endpoint=endpoint,
                status=status,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                tokens_used=tokens_used or (input_tokens + output_tokens),
                estimated_cost_usd=estimated_cost,
                error_message=error_message,
                metadata=metadata or {}
            )

            # Store metrics
            self.metrics.append(metrics)

            # Update session state
            if self.session_state is not None:
                self.session_state['api_metrics'].append(metrics.to_dict())

            # Log to file
            if self.enable_file_logging:
                self._log_to_file(metrics)

            logger.info(
                f"API call recorded: {provider.value}/{endpoint} "
                f"- Status: {status.value}, Duration: {duration_ms:.0f}ms, Cost: ${estimated_cost:.4f}"
            )

            return metrics

    def _calculate_cost(
        self,
        provider: APIProvider,
        tokens_used: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: Optional[str] = None
    ) -> float:
        """Calculate estimated cost for API call."""
        try:
            if provider == APIProvider.BUILD123D:
                return 0.0  # Local operation, free

            elif provider == APIProvider.ZOO_DEV:
                return self.PRICING[provider]['generation']['per_request']

            elif provider in [APIProvider.ANTHROPIC_CLAUDE, APIProvider.ANTHROPIC_VISION]:
                # Use detailed token breakdown if available
                if input_tokens > 0 or output_tokens > 0:
                    # Determine model pricing
                    if model and model in self.PRICING[provider]:
                        pricing = self.PRICING[provider][model]
                    else:
                        # Default to Sonnet pricing
                        pricing = self.PRICING[APIProvider.ANTHROPIC_CLAUDE]['claude-3-5-sonnet-20241022']

                    input_cost = (input_tokens / 1000.0) * pricing['input']
                    output_cost = (output_tokens / 1000.0) * pricing['output']
                    return input_cost + output_cost
                elif tokens_used > 0:
                    # Estimate 50/50 split if only total tokens provided
                    estimated_input = tokens_used * 0.5
                    estimated_output = tokens_used * 0.5
                    pricing = self.PRICING[APIProvider.ANTHROPIC_CLAUDE]['claude-3-5-sonnet-20241022']
                    input_cost = (estimated_input / 1000.0) * pricing['input']
                    output_cost = (estimated_output / 1000.0) * pricing['output']
                    return input_cost + output_cost

            return 0.0

        except Exception as e:
            logger.warning(f"Cost calculation failed: {e}")
            return 0.0

    def _log_to_file(self, metrics: APICallMetrics) -> None:
        """Log metrics to file in JSONL format."""
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log metrics to file: {e}")

    def get_summary_stats(
        self,
        time_window: Optional[timedelta] = None,
        provider: Optional[APIProvider] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for API calls.

        Args:
            time_window: Only include calls within this time window (e.g., timedelta(hours=1))
            provider: Filter by specific provider

        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            # Filter metrics
            filtered_metrics = self.metrics

            if time_window:
                cutoff_time = datetime.now() - time_window
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= cutoff_time]

            if provider:
                filtered_metrics = [m for m in filtered_metrics if m.provider == provider]

            if not filtered_metrics:
                return {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'total_tokens': 0,
                    'total_cost_usd': 0.0,
                    'average_duration_ms': 0.0,
                    'by_provider': {},
                    'by_status': {}
                }

            # Calculate statistics
            total_calls = len(filtered_metrics)
            successful_calls = len([m for m in filtered_metrics if m.status == APICallStatus.SUCCESS])
            failed_calls = total_calls - successful_calls
            total_tokens = sum(m.tokens_used for m in filtered_metrics)
            total_cost = sum(m.estimated_cost_usd for m in filtered_metrics)
            avg_duration = sum(m.duration_ms for m in filtered_metrics) / total_calls

            # Group by provider
            by_provider = {}
            for metric in filtered_metrics:
                p = metric.provider.value
                if p not in by_provider:
                    by_provider[p] = {
                        'calls': 0,
                        'tokens': 0,
                        'cost_usd': 0.0,
                        'avg_duration_ms': 0.0
                    }
                by_provider[p]['calls'] += 1
                by_provider[p]['tokens'] += metric.tokens_used
                by_provider[p]['cost_usd'] += metric.estimated_cost_usd

            # Calculate averages
            for p in by_provider:
                provider_metrics = [m for m in filtered_metrics if m.provider.value == p]
                by_provider[p]['avg_duration_ms'] = sum(m.duration_ms for m in provider_metrics) / len(provider_metrics)

            # Group by status
            by_status = {}
            for metric in filtered_metrics:
                s = metric.status.value
                by_status[s] = by_status.get(s, 0) + 1

            return {
                'total_calls': total_calls,
                'successful_calls': successful_calls,
                'failed_calls': failed_calls,
                'success_rate': successful_calls / total_calls if total_calls > 0 else 0.0,
                'total_tokens': total_tokens,
                'total_cost_usd': total_cost,
                'average_duration_ms': avg_duration,
                'by_provider': by_provider,
                'by_status': by_status,
                'time_window': str(time_window) if time_window else 'all_time'
            }

    def get_recent_calls(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent API calls.

        Args:
            count: Number of recent calls to return

        Returns:
            List of metric dictionaries
        """
        with self._lock:
            recent = sorted(self.metrics, key=lambda m: m.timestamp, reverse=True)[:count]
            return [m.to_dict() for m in recent]

    def get_cost_breakdown(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed cost breakdown by provider and model.

        Returns:
            Nested dictionary: {provider: {model: cost}}
        """
        with self._lock:
            breakdown = {}

            for metric in self.metrics:
                provider = metric.provider.value
                model = metric.metadata.get('model', 'unknown')

                if provider not in breakdown:
                    breakdown[provider] = {}
                if model not in breakdown[provider]:
                    breakdown[provider][model] = 0.0

                breakdown[provider][model] += metric.estimated_cost_usd

            return breakdown

    def export_metrics(self, format: str = 'json') -> str:
        """
        Export all metrics in specified format.

        Args:
            format: Export format ('json' or 'csv')

        Returns:
            Formatted string
        """
        with self._lock:
            if format == 'json':
                return json.dumps([m.to_dict() for m in self.metrics], indent=2)
            elif format == 'csv':
                if not self.metrics:
                    return ""
                # CSV export
                headers = ['timestamp', 'provider', 'endpoint', 'status', 'duration_ms',
                          'tokens_used', 'cost_usd']
                lines = [','.join(headers)]
                for m in self.metrics:
                    line = f"{m.timestamp.isoformat()},{m.provider.value},{m.endpoint}," \
                           f"{m.status.value},{m.duration_ms},{m.tokens_used},{m.estimated_cost_usd}"
                    lines.append(line)
                return '\n'.join(lines)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        with self._lock:
            self.metrics.clear()
            if self.session_state is not None:
                self.session_state['api_metrics'] = []
            logger.info("All metrics cleared")

    def get_provider_usage(self) -> Dict[str, int]:
        """
        Get call count by provider.

        Returns:
            Dictionary mapping provider name to call count
        """
        with self._lock:
            usage = {}
            for metric in self.metrics:
                provider = metric.provider.value
                usage[provider] = usage.get(provider, 0) + 1
            return usage


# Singleton instance for global access
_global_monitor: Optional[APIMonitor] = None


def get_global_monitor() -> APIMonitor:
    """Get or create global API monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = APIMonitor()
    return _global_monitor


def set_global_monitor(monitor: APIMonitor) -> None:
    """Set global API monitor instance."""
    global _global_monitor
    _global_monitor = monitor
