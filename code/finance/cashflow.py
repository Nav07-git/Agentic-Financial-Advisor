from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from data.models import FinancialEvent
from finance.recurrence import (
    RecurringProjection,
    detect_and_project_recurring_patterns,
)


IGNORED_STATUSES = {
    "failed",
    "cancelled",
}

SETTLED_STATUS = "settled"
PENDING_STATUS = "pending"
SCHEDULED_STATUS = "scheduled"

UNREALIZED_STATUS = "unrealized"

DEBIT_DIRECTION = "debit"
CREDIT_DIRECTION = "credit"
NON_CASH_DIRECTION = "non_cash"

INCOME_EVENT_TYPE = "income"
INVESTMENT_VALUATION_EVENT_TYPE = "investment_valuation"

FORECAST_DAYS = 90


@dataclass(frozen=True)
class CashflowEvent:
    """
    A financial event that can affect the cash balance during the forecast.

    This is intentionally separate from FinancialEvent because the forecast
    engine needs a normalized representation of cash-flow behavior.
    """

    event_id: str
    event_date: date
    direction: str
    amount: Decimal | None
    currency: str
    status: str
    event_type: str
    category: str
    requires_amount_resolution: bool = False
    source: str = "explicit"


@dataclass(frozen=True)
class BalancePoint:
    """
    Balance after all cash-flow events on a particular date.
    """

    event_date: date
    starting_balance: Decimal
    inflows: Decimal
    outflows: Decimal
    ending_balance: Decimal


@dataclass(frozen=True)
class ForecastResult:
    """
    Result of a 90-day cash-flow forecast.
    """

    request_date: date
    forecast_end_date: date
    starting_balance: Decimal
    minimum_balance_required: Decimal
    minimum_projected_balance: Decimal
    minimum_balance_date: date
    is_safe: bool
    balance_points: tuple[BalancePoint, ...]


def _normalise(value: str) -> str:
    return value.strip().lower()


def is_cashflow_relevant(event: FinancialEvent) -> bool:
    """
    Determine whether an event is potentially relevant to cash-flow analysis.

    Important challenge rule:
    - failed/cancelled transactions are ignored
    - non-cash events are ignored
    - unrealized investment valuations are ignored
    - pending credits are handled separately and are NOT treated as income
    """

    status = _normalise(event.status)
    direction = _normalise(event.direction)
    event_type = _normalise(event.event_type)

    if status in IGNORED_STATUSES:
        return False

    if direction == NON_CASH_DIRECTION:
        return False

    if (
        event_type == INVESTMENT_VALUATION_EVENT_TYPE
        and status == UNREALIZED_STATUS
    ):
        return False

    return True


def _cashflow_date(event: FinancialEvent) -> date:
    """
    Determine the date on which an event should affect the forecast.

    For future events, a supplied settlement date takes precedence when
    available. Otherwise the event date is used.

    The evidence-resolution layer will eventually be responsible for more
    sophisticated lifecycle/amendment handling.
    """

    return event.settlement_date or event.event_date


def to_cashflow_event(event: FinancialEvent) -> CashflowEvent | None:
    """
    Convert a FinancialEvent into a normalized CashflowEvent.

    This function does not invent missing amounts. Missing amounts remain
    unresolved and must be handled by the image/evidence layer before
    financial calculations depend on them.
    """

    if not is_cashflow_relevant(event):
        return None

    return CashflowEvent(
        event_id=event.event_id,
        event_date=_cashflow_date(event),
        direction=_normalise(event.direction),
        amount=event.amount,
        currency=event.currency.strip().upper(),
        status=_normalise(event.status),
        event_type=_normalise(event.event_type),
        category=event.category.strip().lower(),
        requires_amount_resolution=event.amount is None,
        source="explicit",
    )


def cashflow_treatment(event: CashflowEvent) -> str:
    """
    Classify how an event affects forecast cash.

    Challenge-aligned behavior:

    settled debit
        -> cash outflow

    settled credit
        -> cash inflow

    pending debit
        -> reserved outflow

    pending credit
        -> ignored

    scheduled confirmed income
        -> confirmed inflow

    scheduled non-income credit
        -> ignored unless later evidence establishes it

    scheduled debit
        -> future outflow
    """

    if event.amount is None:
        return "unresolved_amount"

    status = _normalise(event.status)
    direction = _normalise(event.direction)
    event_type = _normalise(event.event_type)

    if status in IGNORED_STATUSES:
        return "ignored"

    if direction == NON_CASH_DIRECTION:
        return "ignored"

    if (
        event_type == INVESTMENT_VALUATION_EVENT_TYPE
        and status == UNREALIZED_STATUS
    ):
        return "ignored"

    if status == SETTLED_STATUS:
        if direction == CREDIT_DIRECTION:
            return "settled_inflow"

        if direction == DEBIT_DIRECTION:
            return "settled_outflow"

    if status == PENDING_STATUS:
        if direction == DEBIT_DIRECTION:
            return "pending_reserved_outflow"

        # HackerRank explicitly says pending credits are ignored.
        if direction == CREDIT_DIRECTION:
            return "ignored"

    if status == SCHEDULED_STATUS:
        if direction == DEBIT_DIRECTION:
            return "scheduled_outflow"

        if (
            direction == CREDIT_DIRECTION
            and event_type == INCOME_EVENT_TYPE
        ):
            return "scheduled_confirmed_inflow"

        # Do not invent future income from arbitrary scheduled credits.
        if direction == CREDIT_DIRECTION:
            return "ignored"

    return "ignored"


def forecast_window(
    request_date: date,
    days: int = FORECAST_DAYS,
) -> tuple[date, date]:
    """
    Return the inclusive 90-day forecast window.

    The challenge defines a 90-day forecast beginning on request_date.
    """

    if days <= 0:
        raise ValueError("Forecast days must be positive.")

    return request_date, request_date + timedelta(days=days)


def events_for_forecast(
    events: list[FinancialEvent],
    user_id: str,
    request_date: date,
    days: int = FORECAST_DAYS,
) -> list[CashflowEvent]:
    """
    Select explicit financial events falling inside the forecast window.

    This function does not project recurring events. That is handled separately
    by recurring_cashflows_for_forecast().
    """

    start_date, end_date = forecast_window(
        request_date=request_date,
        days=days,
    )

    forecast_events: list[CashflowEvent] = []

    for event in events:
        if event.user_id != user_id:
            continue

        cashflow_event = to_cashflow_event(event)

        if cashflow_event is None:
            continue

        if cashflow_event.event_date < start_date:
            continue

        if cashflow_event.event_date > end_date:
            continue

        forecast_events.append(cashflow_event)

    return sorted(
        forecast_events,
        key=lambda event: (
            event.event_date,
            event.event_id,
        ),
    )


def recurring_cashflows_for_forecast(
    events: list[FinancialEvent],
    user_id: str,
    request_date: date,
    days: int = FORECAST_DAYS,
) -> list[CashflowEvent]:
    """
    Detect historical recurring patterns and convert their projections into
    normalized cash-flow events.

    Important:
    these are projections derived from historical recurrence evidence.
    They are not automatically "confirmed" future payments.

    Later, the evidence resolver will determine whether messages/images or
    explicit future records amend, confirm, cancel, or supersede them.
    """

    patterns, projections = detect_and_project_recurring_patterns(
        events=events,
        user_id=user_id,
        request_date=request_date,
        days=days,
    )

    # Keep the local variable because it makes the relationship between
    # detection and projection explicit and easier to inspect/debug.
    _ = patterns

    return [
        _projection_to_cashflow_event(projection)
        for projection in projections
    ]


def _projection_to_cashflow_event(
    projection: RecurringProjection,
) -> CashflowEvent:
    """
    Convert a recurrence projection into the normalized cash-flow model.
    """

    return CashflowEvent(
        event_id=projection.projection_id,
        event_date=projection.event_date,
        direction=projection.direction,
        amount=projection.amount,
        currency=projection.currency,
        status="projected",
        event_type=projection.event_type,
        category=projection.category,
        requires_amount_resolution=False,
        source="recurring_projection",
    )


def _should_include_explicit_event_in_forecast(
    event: CashflowEvent,
    request_date: date,
) -> bool:
    """
    Decide whether an explicit event should affect the forecast.

    The current balance is supplied as the user's available balance on the
    request date. Therefore settled historical transactions before the request
    date must not be applied again.

    Future events, including future scheduled/pending transactions, remain
    candidates for the forecast.
    """

    if event.event_date < request_date:
        return False

    if event.amount is None:
        return False

    treatment = cashflow_treatment(event)

    return treatment != "ignored"


def _treatment_is_inflow(treatment: str) -> bool:
    return treatment in {
        "settled_inflow",
        "scheduled_confirmed_inflow",
    }


def _treatment_is_outflow(treatment: str) -> bool:
    return treatment in {
        "settled_outflow",
        "pending_reserved_outflow",
        "scheduled_outflow",
    }


def _projected_event_is_inflow(event: CashflowEvent) -> bool:
    """
    Recurrence projections preserve their original direction.

    They are treated as candidate projected cash flows, not as confirmed
    income. The forecast currently uses them as recurring historical-pattern
    projections because the challenge explicitly requires recurring income
    and expenses to be considered.
    """

    return event.direction == CREDIT_DIRECTION


def _projected_event_is_outflow(event: CashflowEvent) -> bool:
    return event.direction == DEBIT_DIRECTION


def _build_daily_balance_points(
    starting_balance: Decimal,
    request_date: date,
    forecast_end_date: date,
    events: list[CashflowEvent],
) -> tuple[BalancePoint, ...]:
    """
    Build a daily balance series for the forecast period.

    Using every date rather than only event dates makes the minimum-balance
    check explicit and easy to inspect.
    """

    events_by_date: dict[date, list[CashflowEvent]] = {}

    for event in events:
        if event.event_date < request_date:
            continue

        if event.event_date > forecast_end_date:
            continue

        events_by_date.setdefault(event.event_date, []).append(event)

    balance = starting_balance
    points: list[BalancePoint] = []

    current_date = request_date

    while current_date <= forecast_end_date:
        starting_day_balance = balance
        inflows = Decimal("0")
        outflows = Decimal("0")

        for event in events_by_date.get(current_date, []):
            if event.amount is None:
                continue

            if event.source == "recurring_projection":
                if _projected_event_is_inflow(event):
                    inflows += event.amount
                elif _projected_event_is_outflow(event):
                    outflows += event.amount

                continue

            treatment = cashflow_treatment(event)

            if _treatment_is_inflow(treatment):
                inflows += event.amount
            elif _treatment_is_outflow(treatment):
                outflows += event.amount

        balance = balance + inflows - outflows

        points.append(
            BalancePoint(
                event_date=current_date,
                starting_balance=starting_day_balance,
                inflows=inflows,
                outflows=outflows,
                ending_balance=balance,
            )
        )

        current_date += timedelta(days=1)

    return tuple(points)


def forecast_cash_balance(
    *,
    events: list[FinancialEvent],
    user_id: str,
    request_date: date,
    starting_balance: Decimal,
    minimum_balance: Decimal,
    days: int = FORECAST_DAYS,
    include_recurring: bool = True,
) -> ForecastResult:
    """
    Run the basic 90-day cash-balance forecast.

    This is the deterministic financial foundation for the later affordability
    engine.

    It does NOT:
      - choose a payment method
      - modify flexible expenses
      - create an installment plan
      - decide affordability status
      - calculate amount_safe_to_pay
      - resolve messages/images
      - perform currency conversion

    Those belong to later layers.
    """

    if starting_balance < Decimal("0"):
        raise ValueError("Starting balance cannot be negative.")

    if minimum_balance < Decimal("0"):
        raise ValueError("Minimum balance cannot be negative.")

    start_date, end_date = forecast_window(
        request_date=request_date,
        days=days,
    )

    explicit_events = events_for_forecast(
        events=events,
        user_id=user_id,
        request_date=request_date,
        days=days,
    )

    forecast_events = [
        event
        for event in explicit_events
        if _should_include_explicit_event_in_forecast(
            event=event,
            request_date=request_date,
        )
    ]

    if include_recurring:
        recurring_events = recurring_cashflows_for_forecast(
            events=events,
            user_id=user_id,
            request_date=request_date,
            days=days,
        )

        forecast_events.extend(recurring_events)

    # Deterministic ordering is important for reproducibility and debugging.
    forecast_events.sort(
        key=lambda event: (
            event.event_date,
            event.source,
            event.event_id,
        )
    )

    balance_points = _build_daily_balance_points(
        starting_balance=starting_balance,
        request_date=start_date,
        forecast_end_date=end_date,
        events=forecast_events,
    )

    if not balance_points:
        minimum_projected_balance = starting_balance
        minimum_balance_date = request_date
    else:
        minimum_point = min(
            balance_points,
            key=lambda point: (
                point.ending_balance,
                point.event_date,
            ),
        )

        minimum_projected_balance = minimum_point.ending_balance
        minimum_balance_date = minimum_point.event_date

    is_safe = minimum_projected_balance >= minimum_balance

    return ForecastResult(
        request_date=request_date,
        forecast_end_date=end_date,
        starting_balance=starting_balance,
        minimum_balance_required=minimum_balance,
        minimum_projected_balance=minimum_projected_balance,
        minimum_balance_date=minimum_balance_date,
        is_safe=is_safe,
        balance_points=balance_points,
    )


def summarize_forecast(result: ForecastResult) -> str:
    """
    Produce a compact human-readable diagnostic.

    This is for debugging only. The final decision explanation will be handled
    by a later decision layer.
    """

    status = "SAFE" if result.is_safe else "UNSAFE"

    return (
        f"{status}: "
        f"starting_balance={result.starting_balance}, "
        f"minimum_projected_balance={result.minimum_projected_balance} "
        f"on {result.minimum_balance_date.isoformat()}, "
        f"required_minimum={result.minimum_balance_required}"
    )