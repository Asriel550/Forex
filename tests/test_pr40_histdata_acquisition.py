"""Synthetic local unit tests for HistData downloader, never trading results."""
from io import BytesIO
from zipfile import ZipFile, ZIP_STORED

import numpy as np
import pandas as pd
import pytest

from scripts.acquire_pr40_histdata import (
    SYMBOLS,MONTHS,START,END,
    url_for,parse_source_form,verify_zip,serialize_bars,
)


def test_18_whitelisted_downloads_reject_any_q3_request():
    assert SYMBOLS==("GBPUSD","AUDUSD","USDCAD")
    assert MONTHS==tuple(f"2026{m:02d}" for m in range(1,7))
    assert len({url_for(symbol,month) for symbol in SYMBOLS for month in MONTHS})==18
    assert "gbpusd/2026/1" in url_for("GBPUSD","202601")
    with pytest.raises(ValueError):
        url_for("GBPUSD","202607")
    with pytest.raises(ValueError):
        url_for("EURUSD","202601")
    with pytest.raises(ValueError):
        url_for("USDCAD","202608")


def test_form_requires_pair_and_unique_monthly_m1_ascii_token():
    html='''<form action="/get.php">
     <input name="fxpair" value="AUDUSD">
     <input name="timeframe" value="M1">
     <input name="date" value="2026">
     <input name="datemonth" value="202606">
     <input name="platform" value="ASCII">
     <input name="tk" value="testtoken">
     </form>'''
    values=parse_source_form(html,"AUDUSD","202606")
    assert values["fxpair"]=="AUDUSD" and values["tk"]=="testtoken"
    with pytest.raises(ValueError):
        parse_source_form(html,"GBPUSD","202606")
    with pytest.raises(ValueError):
        parse_source_form(html,"AUDUSD","202607")


def test_zip_validator_rejects_cross_symbol_or_cross_month(tmp_path):
    blob=b"20260105 000000;1.234;1.235;1.230;1.232;0\n"*800
    good=tmp_path/"data.zip"
    with ZipFile(good,"w",compression=ZIP_STORED) as z:
        z.writestr("DAT_ASCII_GBPUSD_M1_202601.csv",blob)
    assert verify_zip(good,"GBPUSD","202601")[0]["uncompressed_bytes"]==len(blob)
    with pytest.raises(ValueError):
        verify_zip(good,"AUDUSD","202601")
    with pytest.raises(ValueError):
        verify_zip(good,"GBPUSD","202602")
    invalid=tmp_path/"nonzip.zip"
    invalid.write_bytes(b"NO DATA"*4000)
    with pytest.raises(ValueError):
        verify_zip(invalid,"GBPUSD","202601")


def test_ohlc_serialization_keeps_open_time_and_price_values():
    idx=pd.date_range("2026-02-05T08:00:00Z",periods=4,freq="5min")
    frame=pd.DataFrame(dict(
        timestamp=idx,open=[1.10]*4,high=[1.11]*4,
        low=[1.09]*4,close=[1.105]*4,volume=[0.]*4
    ))
    result=serialize_bars(frame)
    assert set(result)=={"timestamp_ns","open","high","low","close","volume"}
    assert result["timestamp_ns"].dtype==np.dtype("<i8")
    np.testing.assert_array_equal(
        result["timestamp_ns"],idx.as_unit("ns").asi8
    )
    assert result["close"].tolist()==[1.105]*4


def test_utc_cutoff_is_before_any_q3_candle_engineering():
    ts=pd.DatetimeIndex(["2026-06-30T23:55:00Z",
                         "2026-07-01T00:00:00Z"])
    keep=(ts>=START)&(ts<END)
    assert keep.tolist()==[True,False]


def test_real_histdata_page_has_download_and_status_forms_but_only_zip_is_allowed():
    html='''<form action="/get.php">
    <input name="fxpair" value="GBPUSD">
    <input name="timeframe" value="M1">
    <input name="date" value="2026">
    <input name="datemonth" value="202601">
    <input name="platform" value="ASCII">
    <input name="tk" value="zip_token">
    </form>
    <form action="/getStatus.php">
    <input name="fxpair" value="GBPUSD">
    <input name="timeframe" value="M1">
    <input name="tk" value="status_token">
    </form>'''
    values=parse_source_form(html,"GBPUSD","202601")
    assert values["tk"]=="zip_token"
    html_status_only=html.split("</form>")[1]+"</form>"
    with pytest.raises(ValueError,match="formulario"):
        parse_source_form(html_status_only,"GBPUSD","202601")
