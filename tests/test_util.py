"""Code parsing and the small helpers the scanner depends on."""

from __future__ import annotations

from datetime import timedelta

import pytest

from qr_organizer.util import extract_code, format_code, is_expired, now, parse_code


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BIN-0042", "BIN-0042"),
        ("bin-0042", "BIN-0042"),
        ("  LOC-0001  ", "LOC-0001"),
        ("http://host:8815/s/BIN-0042", "BIN-0042"),
        ("https://box.tailnet.ts.net/s/LOAN-0003", "LOAN-0003"),
        ("http://host/s/BIN-0042?src=camera", "BIN-0042"),
        ("http://host/s/BIN-000042", "BIN-000042"),
    ],
)
def test_codes_are_recovered_from_urls_and_bare_text(raw, expected):
    assert extract_code(raw) == expected


@pytest.mark.parametrize(
    "raw", ["", "hello", "https://example.com/", "BIN", "BIN-", "-0042", "http://host/s/nope"]
)
def test_non_codes_are_rejected_rather_than_guessed(raw):
    assert extract_code(raw) is None


def test_format_and_parse_round_trip():
    assert format_code("bin", 42, 4) == "BIN-0042"
    assert parse_code("BIN-0042") == ("BIN", 42)


def test_expiry_uses_the_configured_window():
    recent = (now() - timedelta(minutes=5)).isoformat()
    stale = (now() - timedelta(minutes=45)).isoformat()
    assert is_expired(recent, 30) is False
    assert is_expired(stale, 30) is True
    assert is_expired(None, 30) is True
