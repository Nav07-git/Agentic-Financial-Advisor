"""Deterministic Stage 2 decision and payment-plan selection.

This module deliberately does not calculate Stage 1 values.  Callers pass
``amount_safe_to_pay`` and ``earliest_date_for_full_payment`` calculated by
``finance.affordability`` and this layer only selects and validates a plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from itertools import combinations
from typing import Iterable

from data.models import FinancialProfile, PaymentOption, Request
from finance.cashflow import ForecastResult


ZERO = Decimal("0")


@dataclass(frozen=True)
class Payment:
    payment_date: date
    amount: Decimal


@dataclass(frozen=True)
class FlexibleExpense:
    """A supported flexible recurring debit in the already-built forecast.

    ``occurrences`` contains each forecast occurrence for this one event.
    The evidence/recurrence layer must supply only recurring, cash-relevant
    events; this module never guesses recurrence from a one-off expense.
    """

    event_id: str
    category: str
    amount: Decimal
    minimum_allowed_amount: Decimal | None
    occurrences: tuple[date, ...]
    is_flexible: bool = True
    can_stop: bool = True
    can_reduce: bool = True


@dataclass(frozen=True)
class SpendingChange:
    event_id: str
    action: str  # "stop" or "reduce_to"
    new_amount: Decimal | None = None

    def serialize(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        assert self.new_amount is not None
        return f"reduce_to:{self.event_id}:{_format_amount(self.new_amount)}"


@dataclass(frozen=True)
class PlanCandidate:
    method: str
    payments: tuple[Payment, ...]
    total_paid: Decimal
    payment_option_id: str | None = None
    spending_changes: tuple[SpendingChange, ...] = ()


@dataclass(frozen=True)
class DecisionResult:
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    spending_changes_needed: str
    candidate: PlanCandidate | None


def _normalise_methods(methods: Iterable[str]) -> set[str]:
    return {method.strip().lower() for method in methods}


def _format_amount(amount: Decimal) -> str:
    """CSV-friendly, non-scientific Decimal formatting."""
    return format(amount, "f")


def serialize_payment_plan(payments: Iterable[Payment]) -> str:
    ordered = sorted(payments, key=lambda payment: payment.payment_date)
    return "|".join(
        f"{payment.payment_date.isoformat()}:{_format_amount(payment.amount)}"
        for payment in ordered
    ) or "none"


def serialize_spending_changes(changes: Iterable[SpendingChange]) -> str:
    items = [change.serialize() for change in changes]
    return "|".join(items) if items else "none"


def payment_option_schedule(option: PaymentOption) -> tuple[Payment, ...]:
    """Return the exact monetary schedule encoded by an installment offer."""
    if option.number_of_payments <= 0:
        return ()
    if option.number_of_payments > 1 and not option.payment_frequency_days:
        return ()

    amounts = [option.payment_amount] * option.number_of_payments
    # Some offers use a rounded final installment.  Preserve total payable
    # exactly rather than introducing a rounding error.
    amounts[-1] = option.total_payable_amount - sum(amounts[:-1], ZERO)
    if any(amount < ZERO for amount in amounts):
        return ()

    frequency = option.payment_frequency_days or 0
    return tuple(
        Payment(option.first_payment_date + timedelta(days=frequency * index), amount)
        for index, amount in enumerate(amounts)
    )


def generate_candidates(
    *, request: Request, profile: FinancialProfile,
    amount_safe_to_pay: Decimal, earliest_date_for_full_payment: date | None,
    payment_options: Iterable[PaymentOption],
) -> list[PlanCandidate]:
    """Generate only rule-supported payment candidates (before safety tests)."""
    methods = _normalise_methods(profile.payment_methods_user_will_consider)
    candidates: list[PlanCandidate] = []

    if "full_payment" in methods:
        candidates.append(PlanCandidate(
            method="full_payment", payments=(Payment(request.request_date, request.requested_amount),),
            total_paid=request.requested_amount,
        ))

    if (
        "partial_payment" in methods and request.allows_partial_payment
        and ZERO < amount_safe_to_pay < request.requested_amount
        and earliest_date_for_full_payment is not None
    ):
        candidates.append(PlanCandidate(
            method="partial_payment",
            payments=(
                Payment(request.request_date, amount_safe_to_pay),
                Payment(earliest_date_for_full_payment, request.requested_amount - amount_safe_to_pay),
            ),
            total_paid=request.requested_amount,
        ))

    if "installments" in methods:
        for option in payment_options:
            if option.request_id != request.request_id:
                continue
            if option.payment_method.strip().lower() != "installments":
                continue
            if (profile.max_installment_months is None
                    or option.number_of_payments > profile.max_installment_months):
                continue
            schedule = payment_option_schedule(option)
            if schedule:
                candidates.append(PlanCandidate(
                    method="installments", payments=schedule,
                    total_paid=option.total_payable_amount,
                    payment_option_id=option.payment_option_id,
                ))

    # Waiting is a recommendation to make the single full payment when it is
    # independently safe, not a newly invented payment option.
    if (
        "full_payment" in methods and earliest_date_for_full_payment is not None
        and earliest_date_for_full_payment > request.request_date
    ):
        candidates.append(PlanCandidate(
            method="wait", payments=(Payment(earliest_date_for_full_payment, request.requested_amount),),
            total_paid=request.requested_amount,
        ))
    return candidates


def candidate_completes_by_deadline(candidate: PlanCandidate, request: Request) -> bool:
    if request.desired_completion_date is None:
        return True
    return bool(candidate.payments) and max(
        payment.payment_date for payment in candidate.payments
    ) <= request.desired_completion_date


def candidate_is_safe(candidate: PlanCandidate, forecast: ForecastResult) -> bool:
    """Apply candidate cash debits to every later daily balance point."""
    by_date: dict[date, Decimal] = {}
    for payment in candidate.payments:
        if payment.amount < ZERO or not (forecast.request_date <= payment.payment_date <= forecast.forecast_end_date):
            return False
        by_date[payment.payment_date] = by_date.get(payment.payment_date, ZERO) + payment.amount

    paid = ZERO
    for point in forecast.balance_points:
        paid += by_date.get(point.event_date, ZERO)
        if point.ending_balance - paid < forecast.minimum_balance_required:
            return False
    return True


def permitted_spending_changes(
    *, profile: FinancialProfile, expenses: Iterable[FlexibleExpense],
) -> tuple[SpendingChange, ...]:
    """Enumerate legal individual changes; combinations are made separately."""
    protected = {item.strip().lower() for item in profile.expense_categories_to_protect}
    reducible = {item.strip().lower() for item in profile.expense_categories_user_is_willing_to_reduce}
    stoppable = {item.strip().lower() for item in profile.expense_categories_user_is_willing_to_stop}
    changes: list[SpendingChange] = []
    for expense in expenses:
        category = expense.category.strip().lower()
        if not expense.is_flexible or category in protected or expense.amount <= ZERO:
            continue
        if expense.can_stop and category in stoppable:
            changes.append(SpendingChange(expense.event_id, "stop"))
        minimum = expense.minimum_allowed_amount
        if (expense.can_reduce and category in reducible and minimum is not None
                and ZERO <= minimum < expense.amount):
            changes.append(SpendingChange(expense.event_id, "reduce_to", minimum))
    return tuple(sorted(changes, key=lambda item: (item.event_id, item.action)))


def apply_spending_changes(
    forecast: ForecastResult, changes: Iterable[SpendingChange],
    expenses: Iterable[FlexibleExpense],
) -> ForecastResult:
    """Return a forecast-equivalent result with known recurring debits changed."""
    expense_by_id = {expense.event_id: expense for expense in expenses}
    benefits: dict[date, Decimal] = {}
    seen: set[str] = set()
    for change in changes:
        if change.event_id in seen or change.event_id not in expense_by_id:
            raise ValueError("Each spending event may be changed at most once.")
        seen.add(change.event_id)
        expense = expense_by_id[change.event_id]
        new_amount = ZERO if change.action == "stop" else change.new_amount
        if new_amount is None or new_amount < ZERO or new_amount > expense.amount:
            raise ValueError("Invalid spending change.")
        for occurrence in expense.occurrences:
            if forecast.request_date <= occurrence <= forecast.forecast_end_date:
                benefits[occurrence] = benefits.get(occurrence, ZERO) + expense.amount - new_amount

    # Import locally to keep the public planning interface independent from
    # cashflow implementation details.
    from finance.cashflow import BalancePoint
    running_benefit = ZERO
    points = []
    for point in forecast.balance_points:
        running_benefit += benefits.get(point.event_date, ZERO)
        points.append(BalancePoint(point.event_date, point.starting_balance + running_benefit,
            point.inflows, point.outflows, point.ending_balance + running_benefit))
    minimum = min(points, key=lambda point: (point.ending_balance, point.event_date))
    return ForecastResult(forecast.request_date, forecast.forecast_end_date,
        forecast.starting_balance, forecast.minimum_balance_required,
        minimum.ending_balance, minimum.event_date,
        minimum.ending_balance >= forecast.minimum_balance_required, tuple(points))


def _rank_key(candidate: PlanCandidate, request: Request) -> tuple:
    # All candidates passed safety.  This is the published priority order.
    return (
        0 if candidate_completes_by_deadline(candidate, request) else 1,
        0 if not candidate.spending_changes else 1,
        candidate.total_paid,
        min(payment.payment_date for payment in candidate.payments),
        len(candidate.payments),
        candidate.payment_option_id or "",
    )


def choose_decision(
    *, request: Request, profile: FinancialProfile, forecast: ForecastResult,
    amount_safe_to_pay: Decimal, earliest_date_for_full_payment: date | None,
    payment_options: Iterable[PaymentOption], expenses: Iterable[FlexibleExpense] = (),
) -> DecisionResult:
    """Select the highest-ranked safe Stage 2 result, including legal changes."""
    base = generate_candidates(request=request, profile=profile,
        amount_safe_to_pay=amount_safe_to_pay,
        earliest_date_for_full_payment=earliest_date_for_full_payment,
        payment_options=payment_options)
    expense_values = tuple(expenses)
    legal_changes = permitted_spending_changes(profile=profile, expenses=expense_values)
    candidates: list[PlanCandidate] = []
    for candidate in base:
        if candidate_completes_by_deadline(candidate, request) and candidate_is_safe(candidate, forecast):
            candidates.append(candidate)
        # Optional changes are only considered after all no-change candidates.
        for size in range(1, min(3, len(legal_changes)) + 1):
            for changes in combinations(legal_changes, size):
                if len({change.event_id for change in changes}) != len(changes):
                    continue
                adjusted = apply_spending_changes(forecast, changes, expense_values)
                changed = PlanCandidate(candidate.method, candidate.payments, candidate.total_paid,
                    candidate.payment_option_id, changes)
                if candidate_completes_by_deadline(changed, request) and candidate_is_safe(changed, adjusted):
                    candidates.append(changed)

    if candidates:
        chosen = min(candidates, key=lambda candidate: _rank_key(candidate, request))
        if chosen.method == "wait":
            return DecisionResult("affordable_later", "wait", serialize_payment_plan(chosen.payments),
                serialize_spending_changes(chosen.spending_changes), chosen)
        status = "affordable_now" if (chosen.method == "full_payment" and chosen.payments[0].payment_date == request.request_date and not chosen.spending_changes) else "affordable_with_plan"
        return DecisionResult(status, chosen.method, serialize_payment_plan(chosen.payments),
            serialize_spending_changes(chosen.spending_changes), chosen)

    accepts_full = "full_payment" in _normalise_methods(
        profile.payment_methods_user_will_consider
    )
    if (
        accepts_full
        and earliest_date_for_full_payment is not None
        and earliest_date_for_full_payment > request.request_date
        and (
            request.desired_completion_date is None
            or earliest_date_for_full_payment <= request.desired_completion_date
        )
    ):
        return DecisionResult("affordable_later", "wait",
            serialize_payment_plan((Payment(earliest_date_for_full_payment, request.requested_amount),)),
            "none", None)
    return DecisionResult("not_affordable", "not_recommended", "none", "none", None)
