"""Retry-After de Telegram; la ingesta estática no usa reintentos HTTP."""
from datetime import datetime,timezone
from src.http_utils import retry_after_seconds


def test_retry_after_numeric_and_date():
    assert retry_after_seconds('3')==3
    assert retry_after_seconds('Fri, 06 Nov 2026 10:00:30 GMT',datetime(2026,11,6,10,0,tzinfo=timezone.utc))==30


def test_invalid_retry_after_has_no_unbounded_wait():
    for value in (None,'nan','inf','invalid','-1'):
        assert retry_after_seconds(value)==0
