"""Utilidades de espera para Telegram; la ingesta de precios no usa HTTP."""
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import math
import time


def retry_after_seconds(value,now=None):
    if value is None:
        return 0.
    try:
        seconds=float(value)
        return max(0.,seconds) if math.isfinite(seconds) else 0.
    except (ValueError,TypeError):
        try:
            deadline=parsedate_to_datetime(value)
            clock=now if now is not None else datetime.now(timezone.utc)
            if deadline.tzinfo is None:deadline=deadline.replace(tzinfo=timezone.utc)
            if clock.tzinfo is None:clock=clock.replace(tzinfo=timezone.utc)
            return max(0.,(deadline-clock).total_seconds())
        except (ValueError,TypeError,OverflowError):
            return 0.


def wait_seconds(seconds):
    while seconds>0:
        part=min(seconds,60.)
        time.sleep(part)
        seconds-=part
