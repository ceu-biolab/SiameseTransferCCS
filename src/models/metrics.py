"""Metrics shared by CCS evaluators."""

from sktime.performance_metrics.forecasting import mean_squared_percentage_error


def mspe_percent(y_true, y_pred) -> float:
    """Return 100 * mean(((y_pred - y_true) / y_true) ** 2).

    This expresses the squared relative error as a percentage, rather than
    squaring errors that have already been converted to percentage points.
    """
    return float(mean_squared_percentage_error(y_true, y_pred) * 100.0)
