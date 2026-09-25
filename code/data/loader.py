from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from .models import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    PaymentOption,
    Request,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def parse_decimal(value: str) -> Decimal | None:
    value = value.strip()

    if not value:
        return None

    return Decimal(value)


def parse_date(value: str) -> date | None:
    value = value.strip()

    if not value:
        return None

    return date.fromisoformat(value)


def parse_datetime(value: str) -> datetime:
    value = value.strip()

    if not value:
        raise ValueError(
            "Expected a timestamp but received an empty value."
        )

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    timestamp = datetime.fromisoformat(value)

    if timestamp.tzinfo is None:
        raise ValueError(
            f"Timestamp must contain timezone information: {value}"
        )

    return timestamp.astimezone(timezone.utc)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Invalid boolean value: {value!r}")


def parse_pipe_list(value: str) -> tuple[str, ...]:
    if not value or not value.strip():
        return ()

    return tuple(
        item.strip()
        for item in value.split("|")
        if item.strip()
    )


def load_requests(dataset_dir: Path) -> list[Request]:
    rows = read_csv(dataset_dir / "requests.csv")

    return [
        Request(
            request_id=row["request_id"],
            user_id=row["user_id"],
            request_date=date.fromisoformat(row["request_date"]),
            request_type=row["request_type"],
            requested_amount=Decimal(row["requested_amount"]),
            desired_completion_date=parse_date(
                row["desired_completion_date"]
            ),
            allows_partial_payment=parse_bool(
                row["allows_partial_payment"]
            ),
            request_text=row["request_text"],
        )
        for row in rows
    ]


def load_profiles(
    dataset_dir: Path,
) -> dict[str, FinancialProfile]:
    rows = read_csv(dataset_dir / "financial_profiles.csv")

    profiles: dict[str, FinancialProfile] = {}

    for row in rows:
        max_months = row["max_installment_months"].strip()

        profiles[row["user_id"]] = FinancialProfile(
            user_id=row["user_id"],
            home_currency=row["home_currency"],
            current_available_balance=Decimal(
                row["current_available_balance"]
            ),
            minimum_balance_to_keep=Decimal(
                row["minimum_balance_to_keep"]
            ),
            financial_priorities=parse_pipe_list(
                row["financial_priorities"]
            ),
            expense_categories_to_protect=parse_pipe_list(
                row["expense_categories_to_protect"]
            ),
            expense_categories_user_is_willing_to_reduce=parse_pipe_list(
                row["expense_categories_user_is_willing_to_reduce"]
            ),
            expense_categories_user_is_willing_to_stop=parse_pipe_list(
                row["expense_categories_user_is_willing_to_stop"]
            ),
            payment_methods_user_will_consider=parse_pipe_list(
                row["payment_methods_user_will_consider"]
            ),
            max_installment_months=(
                int(max_months)
                if max_months
                else None
            ),
        )

    return profiles


def load_financial_events(
    dataset_dir: Path,
) -> list[FinancialEvent]:
    rows = read_csv(dataset_dir / "financial_events.csv")

    return [
        FinancialEvent(
            event_id=row["event_id"],
            user_id=row["user_id"],
            event_type=row["event_type"],
            description=row["description"],
            category=row["category"],
            direction=row["direction"],
            amount=parse_decimal(row["amount"]),
            currency=row["currency"],
            event_date=date.fromisoformat(row["event_date"]),
            settlement_date=parse_date(row["settlement_date"]),
            status=row["status"],
            linked_event_id=row["linked_event_id"] or None,
            flexibility=row["flexibility"] or None,
            minimum_allowed_amount=parse_decimal(
                row["minimum_allowed_amount"]
            ),
        )
        for row in rows
    ]


def load_payment_options(
    dataset_dir: Path,
) -> list[PaymentOption]:
    rows = read_csv(
        dataset_dir / "request_payment_options.csv"
    )

    return [
        PaymentOption(
            payment_option_id=row["payment_option_id"],
            request_id=row["request_id"],
            payment_method=row["payment_method"],
            payment_amount=Decimal(row["payment_amount"]),
            number_of_payments=int(row["number_of_payments"]),
            first_payment_date=date.fromisoformat(
                row["first_payment_date"]
            ),
            payment_frequency_days=(
                int(row["payment_frequency_days"])
                if row["payment_frequency_days"].strip()
                else None
            ),
            financing_fee=Decimal(row["financing_fee"]),
            total_payable_amount=Decimal(
                row["total_payable_amount"]
            ),
        )
        for row in rows
    ]


def load_messages(
    dataset_dir: Path,
) -> list[Message]:
    rows = read_csv(dataset_dir / "messages.csv")

    return [
        Message(
            message_id=row["message_id"],
            user_id=row["user_id"],
            request_id=row["request_id"] or None,
            related_event_id=row["related_event_id"] or None,
            sent_at=parse_datetime(row["sent_at"]),
            source_type=row["source_type"],
            message_text=row["message_text"],
        )
        for row in rows
    ]


def load_images(
    dataset_dir: Path,
) -> list[ImageRecord]:
    rows = read_csv(dataset_dir / "images.csv")

    return [
        ImageRecord(
            image_id=row["image_id"],
            user_id=row["user_id"],
            request_id=row["request_id"] or None,
            related_event_id=row["related_event_id"] or None,
        )
        for row in rows
    ]


def load_exchange_rates(
    dataset_dir: Path,
) -> list[ExchangeRate]:
    rows = read_csv(dataset_dir / "exchange_rates.csv")

    return [
        ExchangeRate(
            rate_date=date.fromisoformat(row["rate_date"]),
            from_currency=row["from_currency"],
            to_currency=row["to_currency"],
            rate=Decimal(row["rate"]),
        )
        for row in rows
    ]