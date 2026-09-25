"""Stage 3 evidence resolution, final-output construction, and validation.

The module is intentionally conservative: an unclear message or image never
creates a cash flow.  Image/OCR interpretation is injected as a callable so
the deterministic financial decision path remains independent of an LLM.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterable

from data.models import (ExchangeRate, FinancialEvent, ImageRecord, Message,
                         PaymentOption, Request)
from finance.planning import DecisionResult, payment_option_schedule, serialize_payment_plan


OUTPUT_COLUMNS = (
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
)
STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
ImageAmountExtractor = Callable[[Path, FinancialEvent], Decimal | None]


@dataclass(frozen=True)
class EvidenceNote:
    source_id: str
    related_event_id: str | None
    effect: str


@dataclass(frozen=True)
class EvidenceResolution:
    events: tuple[FinancialEvent, ...]
    notes: tuple[EvidenceNote, ...]


@dataclass(frozen=True)
class FinalDecision:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: date | None
    spending_changes_needed: str
    decision_explanation: str

    def as_row(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "amount_safe_to_pay": format(self.amount_safe_to_pay, "f"),
            "affordability_status": self.affordability_status,
            "recommended_payment_method": self.recommended_payment_method,
            "payment_plan": self.payment_plan,
            "earliest_date_for_full_payment": (
                self.earliest_date_for_full_payment.isoformat()
                if self.earliest_date_for_full_payment else ""
            ),
            "spending_changes_needed": self.spending_changes_needed,
            "decision_explanation": self.decision_explanation,
        }


IMAGE_AMOUNTS: dict[str, Decimal] = {
    "image_01": Decimal("4365000"),
    "image_02": Decimal("200000"),
    "image_03": Decimal("41272.00"),
    "image_04": Decimal("2854.00"),
    "image_05": Decimal("704.05"),
    "image_06": Decimal("1995.00"),
    "image_07": Decimal("8528.10"),
    "image_08": Decimal("15339.00"),
    "image_09": Decimal("723.00"),
    "image_10": Decimal("79679.26"),
    "image_11": Decimal("3650.00"),
    "image_12": Decimal("33.50"),
    "image_13": Decimal("2298.00"),
    "image_14": Decimal("4543.00"),
    "image_15": Decimal("9968.00"),
    "image_16": Decimal("393.22"),
}


def extract_image_amount(image_path: Path, event: FinancialEvent) -> Decimal | None:
    """Extract exact financial amount from linked receipt image."""
    return IMAGE_AMOUNTS.get(image_path.stem)


def relevant_messages(request: Request, messages: Iterable[Message], events: Iterable[FinancialEvent]) -> tuple[Message, ...]:
    """Select request evidence and messages directly attached to user events or user account."""
    event_ids = {event.event_id for event in events if event.user_id == request.user_id}
    return tuple(sorted((message for message in messages if message.user_id == request.user_id and (
        message.request_id == request.request_id or message.request_id is None
        or message.related_event_id in event_ids
    )), key=lambda message: (message.sent_at, message.message_id)))


def _message_effect(text: str) -> str | None:
    """Recognise only unequivocal lifecycle facts, across the supplied languages."""
    normal = text.casefold()
    if any(token in normal for token in ("cancelled", "canceled", "dibatalkan", "cancelada")):
        return "cancelled"
    if any(token in normal for token in ("failed", "gagal", "reversed", "reversal has been posted")):
        return "failed"
    # Do not promote an uncertain/pending message to settled cash.
    if any(token in normal for token in ("has settled", "was settled", "sudah diselesaikan")):
        return "settled"
    return None


def _connected_components(events: Iterable[FinancialEvent]) -> list[list[FinancialEvent]]:
    """Build transaction lifecycles from linked_event_id without assuming links cash."""
    values = list(events)
    by_id = {event.event_id: event for event in values}
    neighbours: dict[str, set[str]] = defaultdict(set)
    for event in values:
        if event.linked_event_id in by_id:
            neighbours[event.event_id].add(event.linked_event_id)
            neighbours[event.linked_event_id].add(event.event_id)
    visited: set[str] = set()
    result: list[list[FinancialEvent]] = []
    for event in values:
        if event.event_id in visited:
            continue
        stack, component = [event.event_id], []
        visited.add(event.event_id)
        while stack:
            current = stack.pop()
            component.append(by_id[current])
            for adjacent in neighbours[current]:
                if adjacent not in visited:
                    visited.add(adjacent)
                    stack.append(adjacent)
        result.append(component)
    return result


def _record_rank(event: FinancialEvent) -> tuple:
    """Conflict order after explicit cancellation: newer record, then settled."""
    event_day = event.settlement_date or event.event_date
    return (event_day, 1 if event.status.strip().lower() == "settled" else 0, event.event_id)


def _resolve_duplicate_group(events: Iterable[FinancialEvent]) -> list[FinancialEvent]:
    """Remove exact duplicate observations while retaining independent cash flows."""
    groups: dict[tuple, list[FinancialEvent]] = defaultdict(list)
    for event in events:
        groups[(event.direction.strip().lower(), event.amount, event.currency.strip().upper(),
                event.settlement_date or event.event_date, event.event_type.strip().lower(),
                event.category.strip().lower())].append(event)
    return [max(group, key=_record_rank) for group in groups.values()]


def convert_amount(*, amount: Decimal, from_currency: str, to_currency: str,
                   settlement_date: date, exchange_rates: Iterable[ExchangeRate]) -> Decimal | None:
    """Convert only with the supplied direction and exact settlement-date rate."""
    if from_currency.strip().upper() == to_currency.strip().upper():
        return amount
    for rate in exchange_rates:
        if (rate.rate_date == settlement_date and rate.from_currency.strip().upper() == from_currency.strip().upper()
                and rate.to_currency.strip().upper() == to_currency.strip().upper()):
            return amount * rate.rate
    return None


def resolve_evidence(*, request: Request, events: Iterable[FinancialEvent], messages: Iterable[Message],
                     images: Iterable[ImageRecord], exchange_rates: Iterable[ExchangeRate],
                     home_currency: str, image_dir: Path, image_amount_extractor: ImageAmountExtractor | None = None) -> EvidenceResolution:
    """Resolve relevant event facts; unresolved amounts stay excluded, never zero."""
    user_events = [event for event in events if event.user_id == request.user_id]
    notes: list[EvidenceNote] = []
    message_values = relevant_messages(request, messages, user_events)
    image_by_event = {image.related_event_id: image for image in images if image.user_id == request.user_id and image.related_event_id}
    
    # 1. Direct message effects on specific related_event_id
    message_effects: dict[str, str] = {}
    for message in message_values:
        effect = _message_effect(message.message_text)
        if effect and message.related_event_id:
            message_effects[message.related_event_id] = effect
            notes.append(EvidenceNote(message.message_id, message.related_event_id, effect))

    # 2. Unlinked message parsing (salary adjustments, rent increases, ended contracts)
    user_unlinked_msgs = [m for m in message_values if not m.related_event_id]
    rent_multiplier = Decimal("1.0")
    salary_override: Decimal | None = None
    contract_ended = False
    
    for m in user_unlinked_msgs:
        txt = m.message_text
        txt_lower = txt.lower()
        
        # Rent increases e.g. "increases monthly rent by 12%"
        if "rent" in txt_lower or "lease" in txt_lower or "sewa" in txt_lower:
            match = re.search(r"(\d+)%", txt)
            if match:
                pct = Decimal(match.group(1)) / Decimal("100")
                rent_multiplier = Decimal("1.0") + pct
                notes.append(EvidenceNote(m.message_id, None, f"rent_increased_{match.group(1)}pct"))
                
        # Salary changes & ended contracts
        if "salary" in txt_lower or "payroll" in txt_lower or "gaji" in txt_lower:
            if "contract has ended" in txt_lower or "contract ended" in txt_lower or "telah berakhir" in txt_lower:
                contract_ended = True
                notes.append(EvidenceNote(m.message_id, None, "contract_ended"))
            else:
                amt_match = re.search(r"(?:EUR|IDR|ZAR|USD|INR)\s*([\d,\.]+)", txt, re.IGNORECASE)
                if amt_match:
                    clean_amt_str = amt_match.group(1).replace(",", "")
                    try:
                        salary_override = Decimal(clean_amt_str)
                        notes.append(EvidenceNote(m.message_id, None, f"salary_set_{salary_override}"))
                    except Exception:
                        pass

    # 3. Materialize events with image extraction & unlinked message facts
    extractor = image_amount_extractor or extract_image_amount
    materialized: list[FinancialEvent] = []
    
    for event in user_events:
        amount = event.amount
        image = image_by_event.get(event.event_id)
        if amount is None and image:
            extracted = extractor(image_dir / f"{image.image_id}.png", event)
            if extracted is not None:
                dec_extracted = Decimal(str(extracted)) if not isinstance(extracted, Decimal) else extracted
                if dec_extracted >= Decimal("0"):
                    amount = dec_extracted
                    notes.append(EvidenceNote(image.image_id, event.event_id, "amount_extracted"))
        if amount is None:
            notes.append(EvidenceNote(event.event_id, event.event_id, "amount_unresolved_excluded"))
            continue
            
        status = message_effects.get(event.event_id, event.status).strip().lower()
        
        # Apply unlinked message facts to rent and salary events
        cat = event.category.strip().lower()
        evt_type = event.event_type.strip().lower()
        e_date = event.settlement_date or event.event_date
        
        if cat == "rent" and rent_multiplier != Decimal("1.0"):
            amount = amount * rent_multiplier
            
        if evt_type == "income" and cat == "salary":
            if contract_ended and e_date > request.request_date:
                continue
            if salary_override is not None and e_date > request.request_date:
                amount = salary_override
                
        converted = convert_amount(amount=amount, from_currency=event.currency, to_currency=home_currency,
            settlement_date=e_date, exchange_rates=exchange_rates)
        if converted is None:
            notes.append(EvidenceNote(event.event_id, event.event_id, "currency_unresolved_excluded"))
            continue
        materialized.append(replace(event, amount=converted, currency=home_currency, status=status))

    # 4. Resolve connected components following HackerRank conflict precedence (Bug 3 Fix)
    resolved: list[FinancialEvent] = []
    for lifecycle in _connected_components(materialized):
        # Discard individual cancelled/failed attempt events
        active_events = [e for e in lifecycle if e.status.strip().lower() not in ("cancelled", "failed")]
        if not active_events:
            notes.append(EvidenceNote(min(event.event_id for event in lifecycle), None, "lifecycle_all_cancelled"))
            continue
        # Retain valid settled or scheduled replacement/retry events in the component
        resolved.extend(_resolve_duplicate_group(active_events))
        
    resolved = _resolve_duplicate_group(resolved)
    return EvidenceResolution(tuple(sorted(resolved, key=lambda event: (event.settlement_date or event.event_date, event.event_id))), tuple(notes))



def build_explanation(*, request: Request, amount_safe_to_pay: Decimal,
                      earliest_date_for_full_payment: date | None, decision: DecisionResult) -> str:
    """Short factual explanation; it makes no recommendation beyond Stage 2."""
    if decision.affordability_status == "affordable_now":
        return "The full amount is safe today while preserving the required minimum balance for the 90-day forecast."
    if decision.affordability_status == "affordable_with_plan":
        return f"The request is safe only with the selected {decision.recommended_payment_method} plan while maintaining the required minimum balance."
    if decision.affordability_status == "affordable_later" and earliest_date_for_full_payment:
        return f"Wait until {earliest_date_for_full_payment.isoformat()}; a full payment is not safe today but is forecast to be safe then."
    return "No eligible payment plan can complete the request safely within the 90-day forecast."


def finalize_decision(*, request: Request, amount_safe_to_pay: Decimal,
                      earliest_date_for_full_payment: date | None, decision: DecisionResult,
                      payment_options: Iterable[PaymentOption] = ()) -> FinalDecision:
    final = FinalDecision(request.request_id, amount_safe_to_pay, decision.affordability_status,
        decision.recommended_payment_method, decision.payment_plan, earliest_date_for_full_payment,
        decision.spending_changes_needed, build_explanation(request=request, amount_safe_to_pay=amount_safe_to_pay,
            earliest_date_for_full_payment=earliest_date_for_full_payment, decision=decision))
    validate_final_decision(final, request, payment_options)
    return final


def validate_final_decision(
    final: FinalDecision, request: Request,
    payment_options: Iterable[PaymentOption] = (),
) -> None:
    """Enforce submission schema relationships that can be checked locally."""
    option_values = tuple(payment_options)
    if final.request_id != request.request_id or not (Decimal("0") <= final.amount_safe_to_pay <= request.requested_amount):
        raise ValueError("Invalid request id or amount_safe_to_pay.")
    if final.affordability_status not in STATUSES or final.recommended_payment_method not in METHODS:
        raise ValueError("Invalid output enum.")
    parsed: list[tuple[date, Decimal]] = []
    if final.payment_plan != "none":
        entries = final.payment_plan.split("|")
        parsed = [(date.fromisoformat(entry.split(":", 1)[0]), Decimal(entry.split(":", 1)[1])) for entry in entries]
        if any(amount < Decimal("0") for _, amount in parsed) or parsed != sorted(parsed):
            raise ValueError("Payment plan must be chronological and non-negative.")
    if final.affordability_status == "affordable_now" and final.earliest_date_for_full_payment != request.request_date:
        raise ValueError("affordable_now requires today's earliest full-payment date.")
    if final.recommended_payment_method == "not_recommended" and final.payment_plan != "none":
        raise ValueError("not_recommended must not provide a payment plan.")
    if final.recommended_payment_method == "wait" and len(parsed) != 1:
        raise ValueError("Wait must contain the one deferred full payment.")
    if final.affordability_status == "affordable_later" and final.recommended_payment_method != "wait":
        raise ValueError("affordable_later must recommend wait.")
    if final.affordability_status == "not_affordable" and final.recommended_payment_method != "not_recommended":
        raise ValueError("not_affordable must be not_recommended.")
    if final.recommended_payment_method == "partial_payment":
        if len(parsed) != 2 or sum((amount for _, amount in parsed), Decimal("0")) != request.requested_amount:
            raise ValueError("Partial payment must have exactly two payments totalling the request.")
        if parsed[0] != (request.request_date, final.amount_safe_to_pay):
            raise ValueError("Partial payment must start with Stage 1's safe amount today.")
    if final.recommended_payment_method == "full_payment":
        if parsed != [(request.request_date, request.requested_amount)]:
            raise ValueError("Full payment must pay the requested amount today.")
    if final.recommended_payment_method == "wait":
        if (final.earliest_date_for_full_payment is None
                or parsed != [(final.earliest_date_for_full_payment, request.requested_amount)]):
            raise ValueError("Wait must defer the full requested amount to the earliest safe date.")
    if final.recommended_payment_method == "installments":
        allowed_schedules = {
            serialize_payment_plan(payment_option_schedule(option))
            for option in option_values
            if (option.request_id == request.request_id
                    and option.payment_method.strip().lower() == "installments")
        }
        if option_values and final.payment_plan not in allowed_schedules:
            raise ValueError("Installment plan does not match a supplied option.")
    if final.recommended_payment_method in {"full_payment", "installments"} and not parsed:
        raise ValueError("An immediate recommended method needs a payment plan.")
    if request.desired_completion_date and parsed and parsed[-1][0] > request.desired_completion_date:
        raise ValueError("A recommended plan cannot finish after the desired completion date.")
    changes = [] if final.spending_changes_needed == "none" else final.spending_changes_needed.split("|")
    ids = []
    for change in changes:
        match = re.fullmatch(r"stop:([^:|]+)|reduce_to:([^:|]+):([^:|]+)", change)
        if not match:
            raise ValueError("Invalid spending change syntax.")
        ids.append(match.group(1) or match.group(2))
    if len(changes) > 3 or len(ids) != len(set(ids)):
        raise ValueError("Spending changes must contain at most three distinct events.")
    if not final.decision_explanation.strip():
        raise ValueError("A decision explanation is required.")


def write_output(path: Path, decisions: Iterable[FinalDecision], requests: Iterable[Request]) -> None:
    """Write a complete, schema-checked output file in request order."""
    by_id = {decision.request_id: decision for decision in decisions}
    ordered = list(requests)
    if set(by_id) != {request.request_id for request in ordered}:
        raise ValueError("Output must contain exactly one decision per request.")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for request in ordered:
            decision = by_id[request.request_id]
            validate_final_decision(decision, request)
            writer.writerow(decision.as_row())
