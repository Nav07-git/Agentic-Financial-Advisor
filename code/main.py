"""Runnable end-to-end generator for the Buy or Wait? submission."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from agent_tools import extract_receipt_amount
from data.loader import (load_exchange_rates, load_financial_events, load_images,
                         load_messages, load_payment_options, load_profiles, load_requests)
from finance.affordability import (calculate_amount_safe_to_pay,
                                   calculate_earliest_date_for_full_payment)
from finance.cashflow import forecast_cash_balance
from finance.evidence import (finalize_decision, resolve_evidence, write_output)
from finance.planning import FlexibleExpense, choose_decision
from finance.recurrence import detect_and_project_recurring_patterns
from tracker import UsageTracker


def _flexible_expenses(events, user_id, request_date):
    """Map only demonstrated recurring flexible debit projections to Stage 2."""
    _, projections = detect_and_project_recurring_patterns(
        events=events, user_id=user_id, request_date=request_date,
    )
    grouped, metadata = defaultdict(list), {}
    for projection in projections:
        flexibility = (projection.flexibility or "").strip().lower()
        if projection.direction != "debit" or flexibility not in {
            "stoppable", "reducible", "reducible_or_stoppable"
        }:
            continue
        grouped[projection.source_event_id].append(projection.event_date)
        metadata[projection.source_event_id] = projection
    return tuple(
        FlexibleExpense(
            event_id=source_id, category=projection.category,
            amount=projection.amount,
            minimum_allowed_amount=projection.minimum_allowed_amount,
            occurrences=tuple(sorted(grouped[source_id])), is_flexible=True,
            can_stop=(projection.flexibility or "").strip().lower()
            in {"stoppable", "reducible_or_stoppable"},
            can_reduce=(projection.flexibility or "").strip().lower()
            in {"reducible", "reducible_or_stoppable"},
        )
        for source_id, projection in sorted(metadata.items())
    )


def run(project_root: Path | None = None) -> Path:
    project_root = project_root or Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"
    requests = load_requests(dataset_dir)
    profiles = load_profiles(dataset_dir)
    events = load_financial_events(dataset_dir)
    messages = load_messages(dataset_dir)
    images = load_images(dataset_dir)
    rates = load_exchange_rates(dataset_dir)
    options = load_payment_options(dataset_dir)
    output = []

    tracker = UsageTracker.get_instance()

    for request in requests:
        tracker.record_request()
        profile = profiles[request.user_id]
        evidence = resolve_evidence(
            request=request, events=events, messages=messages, images=images,
            exchange_rates=rates, home_currency=profile.home_currency,
            image_dir=dataset_dir / "media" / "images",
            image_amount_extractor=extract_receipt_amount,
        )

        forecast = forecast_cash_balance(
            events=list(evidence.events), user_id=request.user_id,
            request_date=request.request_date,
            starting_balance=profile.current_available_balance,
            minimum_balance=profile.minimum_balance_to_keep,
        )
        amount_safe = calculate_amount_safe_to_pay(
            requested_amount=request.requested_amount, forecast=forecast)
        earliest = calculate_earliest_date_for_full_payment(
            requested_amount=request.requested_amount, forecast=forecast)
        decision = choose_decision(
            request=request, profile=profile, forecast=forecast,
            amount_safe_to_pay=amount_safe, earliest_date_for_full_payment=earliest,
            payment_options=options,
            expenses=_flexible_expenses(list(evidence.events), request.user_id,
                                        request.request_date),
        )
        output.append(finalize_decision(
            request=request, amount_safe_to_pay=amount_safe,
            earliest_date_for_full_payment=earliest, decision=decision,
            payment_options=options))

    output_path = project_root / "output.csv"
    write_output(output_path, output, requests)

    report_path = project_root / "evaluation" / "usage_report.md"
    tracker.write_report(report_path)

    return output_path



if __name__ == "__main__":
    print(run())
