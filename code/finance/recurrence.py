from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from statistics import median
import re

from data.models import FinancialEvent


MIN_OCCURRENCES = 3

MAX_INTERVAL_RELATIVE_DEVIATION = Decimal("0.20")

MONTHLY_INTERVAL_MIN = 27
MONTHLY_INTERVAL_MAX = 32

BI_MONTHLY_INTERVAL_MIN = 55
BI_MONTHLY_INTERVAL_MAX = 65

SETTLED_STATUS = "settled"
IGNORED_STATUSES = {"failed", "cancelled"}

DEBIT_DIRECTION = "debit"
CREDIT_DIRECTION = "credit"
NON_CASH_DIRECTION = "non_cash"

UNREALIZED_STATUS = "unrealized"
INVESTMENT_VALUATION_EVENT_TYPE = "investment_valuation"


@dataclass(frozen=True)
class RecurringPattern:
    user_id: str
    direction: str
    event_type: str
    category: str
    currency: str
    description: str
    interval_days: int
    amount: Decimal
    anchor_date: date
    flexibility: str | None
    minimum_allowed_amount: Decimal | None
    source_event_ids: tuple[str, ...]
    recurrence_type: str = "interval"
    day_of_month: int | None = None
    month_step: int | None = None

    @property
    def representative_amount(self) -> Decimal:
        return self.amount


@dataclass(frozen=True)
class RecurringProjection:
    projection_id: str
    source_event_id: str
    event_date: date
    direction: str
    amount: Decimal
    currency: str
    event_type: str
    category: str
    flexibility: str | None
    minimum_allowed_amount: Decimal | None


def _normalise_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)
    return value


def _is_historical_candidate(event: FinancialEvent) -> bool:
    status = event.status.strip().lower()
    direction = event.direction.strip().lower()
    event_type = event.event_type.strip().lower()

    if status in IGNORED_STATUSES:
        return False

    if status != SETTLED_STATUS:
        return False

    if direction == NON_CASH_DIRECTION:
        return False

    if (
        event_type == INVESTMENT_VALUATION_EVENT_TYPE
        and status == UNREALIZED_STATUS
    ):
        return False

    if event.amount is None:
        return False

    if event.amount <= Decimal("0"):
        return False

    return True


def _group_key(event: FinancialEvent) -> tuple[str, ...]:
    return (
        event.direction.strip().lower(),
        event.event_type.strip().lower(),
        event.category.strip().lower(),
        event.currency.strip().upper(),
        _normalise_text(event.description),
    )


def _median_decimal(values: list[Decimal]) -> Decimal:
    if not values:
        raise ValueError(
            "Cannot calculate median of an empty amount list."
        )

    return Decimal(str(median(values)))


def _infer_interval_days(events: list[FinancialEvent]) -> int:
    dates = sorted(event.event_date for event in events)

    gaps = [
        (later - earlier).days
        for earlier, later in zip(dates, dates[1:])
    ]

    if not gaps:
        raise ValueError("At least two dates are required.")

    return max(1, int(round(median(gaps))))


def _timing_is_consistent(
    events: list[FinancialEvent],
) -> bool:
    dates = sorted(event.event_date for event in events)

    gaps = [
        (later - earlier).days
        for earlier, later in zip(dates, dates[1:])
    ]

    if not gaps:
        return False

    representative = Decimal(str(median(gaps)))

    if representative <= 0:
        return False

    for gap in gaps:
        difference = abs(
            Decimal(gap) - representative
        )

        relative_difference = (
            difference / representative
        )

        if (
            relative_difference
            > MAX_INTERVAL_RELATIVE_DEVIATION
        ):
            return False

    return True


def _infer_calendar_month_step(
    events: list[FinancialEvent],
) -> int | None:
    """
    Detect recurring schedules based on calendar months.

    Examples:

        Mar 8, Apr 8, May 8
            -> step = 1

        Mar 8, May 8, Jul 8
            -> step = 2

    We require the same day-of-month and a consistent number of
    calendar months between observations.
    """

    if len(events) < MIN_OCCURRENCES:
        return None

    ordered = sorted(
        events,
        key=lambda event: (
            event.event_date,
            event.event_id,
        ),
    )

    dates = [event.event_date for event in ordered]

    if len({event_date.day for event_date in dates}) != 1:
        return None

    month_positions = [
        event_date.year * 12 + event_date.month
        for event_date in dates
    ]

    month_gaps = [
        later - earlier
        for earlier, later in zip(
            month_positions,
            month_positions[1:],
        )
    ]

    if not month_gaps:
        return None

    representative = int(round(median(month_gaps)))

    if representative <= 0:
        return None

    if any(
        gap != representative
        for gap in month_gaps
    ):
        return None

    return representative


def _resolve_flexibility(
    events: list[FinancialEvent],
) -> tuple[str | None, Decimal | None]:

    flexibility_values = [
        event.flexibility.strip()
        for event in events
        if (
            event.flexibility is not None
            and event.flexibility.strip()
        )
    ]

    flexibility = None

    if flexibility_values:
        non_fixed = [
            value
            for value in flexibility_values
            if value.lower() != "fixed"
        ]

        flexibility = (
            non_fixed[-1]
            if non_fixed
            else flexibility_values[-1]
        )

    minimum_values = [
        event.minimum_allowed_amount
        for event in events
        if (
            event.minimum_allowed_amount is not None
            and event.minimum_allowed_amount
            >= Decimal("0")
        )
    ]

    minimum_allowed_amount = (
        min(minimum_values)
        if minimum_values
        else None
    )

    return flexibility, minimum_allowed_amount


def _projection_date_monthly(
    year: int,
    month: int,
    day_of_month: int,
) -> date:

    last_day = monthrange(year, month)[1]

    day = min(
        day_of_month,
        last_day,
    )

    return date(
        year,
        month,
        day,
    )


def _add_months(
    source_date: date,
    months: int,
    day_of_month: int,
) -> date:

    zero_based_month = (
        source_date.month - 1 + months
    )

    year = (
        source_date.year
        + zero_based_month // 12
    )

    month = (
        zero_based_month % 12
    ) + 1

    return _projection_date_monthly(
        year=year,
        month=month,
        day_of_month=day_of_month,
    )


def detect_recurring_patterns(
    events: list[FinancialEvent],
    user_id: str | None = None,
) -> list[RecurringPattern]:

    grouped: dict[
        tuple[str, ...],
        list[FinancialEvent],
    ] = {}

    for event in events:

        if (
            user_id is not None
            and event.user_id != user_id
        ):
            continue

        if not _is_historical_candidate(event):
            continue

        key = _group_key(event)

        grouped.setdefault(
            key,
            [],
        ).append(event)

    patterns: list[RecurringPattern] = []

    for grouped_events in grouped.values():

        if len(grouped_events) < MIN_OCCURRENCES:
            continue

        ordered_events = sorted(
            grouped_events,
            key=lambda event: (
                event.event_date,
                event.event_id,
            ),
        )

        if not _timing_is_consistent(
            ordered_events
        ):
            continue

        interval_days = _infer_interval_days(
            ordered_events
        )

        representative_amount = _median_decimal(
            [
                event.amount
                for event in ordered_events
                if event.amount is not None
            ]
        )

        anchor_event = ordered_events[-1]

        flexibility, minimum_allowed_amount = (
            _resolve_flexibility(
                ordered_events
            )
        )

        month_step = _infer_calendar_month_step(
            ordered_events
        )

        if month_step is not None:
            recurrence_type = "calendar_monthly"
            day_of_month = anchor_event.event_date.day
        else:
            recurrence_type = "interval"
            day_of_month = None

        pattern = RecurringPattern(
            user_id=anchor_event.user_id,
            direction=anchor_event.direction.strip().lower(),
            event_type=anchor_event.event_type.strip().lower(),
            category=anchor_event.category.strip().lower(),
            currency=anchor_event.currency.strip().upper(),
            description=anchor_event.description.strip(),
            interval_days=interval_days,
            amount=representative_amount,
            anchor_date=anchor_event.event_date,
            flexibility=flexibility,
            minimum_allowed_amount=minimum_allowed_amount,
            source_event_ids=tuple(
                event.event_id
                for event in ordered_events
            ),
            recurrence_type=recurrence_type,
            day_of_month=day_of_month,
            month_step=month_step,
        )

        patterns.append(pattern)

    return sorted(
        patterns,
        key=lambda pattern: (
            pattern.user_id,
            pattern.anchor_date,
            pattern.direction,
            pattern.category,
            pattern.description.lower(),
        ),
    )


def project_recurring_patterns(
    patterns: list[RecurringPattern],
    request_date: date,
    days: int = 90,
) -> list[RecurringProjection]:

    if days <= 0:
        raise ValueError(
            "Projection window must contain at least one day."
        )

    end_date = request_date + timedelta(days=days)

    projections: list[RecurringProjection] = []

    for pattern in patterns:

        if (
            pattern.recurrence_type
            == "calendar_monthly"
        ):

            if (
                pattern.day_of_month is None
                or pattern.month_step is None
            ):
                continue

            month_step = pattern.month_step

            months_from_anchor = 0

            projected_date = _add_months(
                source_date=pattern.anchor_date,
                months=months_from_anchor,
                day_of_month=pattern.day_of_month,
            )

            while projected_date < request_date:
                months_from_anchor += month_step

                projected_date = _add_months(
                    source_date=pattern.anchor_date,
                    months=months_from_anchor,
                    day_of_month=pattern.day_of_month,
                )

            while projected_date <= end_date:

                projections.append(
                    _make_projection(
                        pattern=pattern,
                        event_date=projected_date,
                    )
                )

                months_from_anchor += month_step

                projected_date = _add_months(
                    source_date=pattern.anchor_date,
                    months=months_from_anchor,
                    day_of_month=pattern.day_of_month,
                )

        else:

            projected_date = pattern.anchor_date

            while projected_date < request_date:
                projected_date += timedelta(
                    days=pattern.interval_days
                )

            while projected_date <= end_date:

                projections.append(
                    _make_projection(
                        pattern=pattern,
                        event_date=projected_date,
                    )
                )

                projected_date += timedelta(
                    days=pattern.interval_days
                )

    return sorted(
        projections,
        key=lambda projection: (
            projection.event_date,
            projection.projection_id,
        ),
    )


def _make_projection(
    pattern: RecurringPattern,
    event_date: date,
) -> RecurringProjection:

    projection_id = (
        f"recurring:"
        f"{pattern.user_id}:"
        f"{pattern.direction}:"
        f"{pattern.category}:"
        f"{_normalise_text(pattern.description)}:"
        f"{event_date.isoformat()}"
    )

    return RecurringProjection(
        projection_id=projection_id,
        source_event_id=pattern.source_event_ids[-1],
        event_date=event_date,
        direction=pattern.direction,
        amount=pattern.amount,
        currency=pattern.currency,
        event_type=pattern.event_type,
        category=pattern.category,
        flexibility=pattern.flexibility,
        minimum_allowed_amount=pattern.minimum_allowed_amount,
    )


def project_recurring_events(
    patterns: list[RecurringPattern],
    request_date: date,
    days: int = 90,
) -> list[RecurringProjection]:

    return project_recurring_patterns(
        patterns=patterns,
        request_date=request_date,
        days=days,
    )


def detect_and_project_recurring_patterns(
    events: list[FinancialEvent],
    user_id: str,
    request_date: date,
    days: int = 90,
) -> tuple[
    list[RecurringPattern],
    list[RecurringProjection],
]:

    patterns = detect_recurring_patterns(
        events=events,
        user_id=user_id,
    )

    projections = project_recurring_patterns(
        patterns=patterns,
        request_date=request_date,
        days=days,
    )

    return patterns, projections