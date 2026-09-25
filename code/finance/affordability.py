from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from finance.cashflow import ForecastResult


@dataclass(frozen=True)
class AffordabilityResult:
    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: date | None


def calculate_amount_safe_to_pay(
    *,
    requested_amount: Decimal,
    forecast: ForecastResult,
) -> Decimal:
    """
    Calculate the largest amount that can be paid on the request date
    without causing the 90-day forecast to fall below the required
    minimum balance.

    HackerRank requirement:
    amount_safe_to_pay is calculated before optional spending changes.
    """
    if requested_amount < Decimal("0"):
        raise ValueError("Requested amount cannot be negative.")

    available_capacity = (
        forecast.minimum_projected_balance
        - forecast.minimum_balance_required
    )

    if available_capacity < Decimal("0"):
        available_capacity = Decimal("0")

    return min(requested_amount, available_capacity)


def calculate_earliest_date_for_full_payment(
    *,
    requested_amount: Decimal,
    forecast: ForecastResult,
) -> date | None:
    """
    Find the earliest date on which the full requested amount can be
    paid as a single payment while keeping the balance at or above the
    required minimum throughout the remaining forecast period.

    This calculation is independent of payment-method preferences.
    """
    if requested_amount < Decimal("0"):
        raise ValueError("Requested amount cannot be negative.")

    if requested_amount == Decimal("0"):
        return forecast.request_date

    minimum_balance = forecast.minimum_balance_required

    # The balance after a full payment on a candidate date must remain
    # at or above the required minimum from that date onward.
    balance_points = forecast.balance_points

    for index, point in enumerate(balance_points):
        if point.event_date < forecast.request_date:
            continue

        balance_after_payment = (
            point.ending_balance - requested_amount
        )

        if balance_after_payment < minimum_balance:
            continue

        remaining_points = balance_points[index + 1 :]

        safe_after_payment = all(
            remaining_point.ending_balance - requested_amount
            >= minimum_balance
            for remaining_point in remaining_points
        )

        if safe_after_payment:
            return point.event_date

    return None