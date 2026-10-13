"""Generate data.json for the existing Stocks page.

Restored from update.pdf; the original scoring rules are preserved.
Educational analysis only. This script does not place trades.
"""

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD",
    "AVGO", "NFLX", "JPM", "V", "COST", "LLY", "CRM", "ORCL",
]


def rsi(c, n=14):
    d = c.diff()
    u = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    w = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    result = 100 - 100 / (1 + u / w)
    return result.mask((u == 0) & (w == 0), 50)


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"], (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd_hist(c):
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    return m - m.ewm(span=9, adjust=False).mean()


def get(sym):
    import yfinance as yf
    df = yf.Ticker(sym).history(period="2y", auto_adjust=True, timeout=20)
    if df.empty:
        raise RuntimeError(f"No price history for {sym}")
    return df.dropna(subset=["Close", "High", "Low", "Volume"]).sort_index()


def analyze(sym, df, bench):
    if len(df) < 210:
        return None
    c = df["Close"]
    s20, s50, s200 = (c.rolling(n).mean() for n in (20, 50, 200))
    r, a, h = rsi(c), atr(df), macd_hist(c)
    px, A = float(c.iloc[-1]), float(a.iloc[-1])
    rs = float(r.iloc[-1])
    if not all(math.isfinite(v) for v in (px, A, rs, float(bench))):
        return None
    if px <= 0 or A <= 0:
        return None
    ret = px / c.iloc[-126] - 1
    avg_volume = float(df["Volume"].rolling(50).mean().iloc[-1])
    vr = float(df["Volume"].iloc[-1]) / avg_volume if avg_volume > 0 else 0
    dist = px / c.iloc[-252:].max() - 1
    ext = (px - s20.iloc[-1]) / A
    sc, why = 0, []
    if px > s200.iloc[-1]:
        sc += 20
        why.append("فوق SMA200")
    if s50.iloc[-1] > s200.iloc[-1]:
        sc += 15
        why.append("اتجاه صاعد")
    if px > s50.iloc[-1]:
        sc += 10
    if h.iloc[-1] > 0 and h.iloc[-1] > h.iloc[-2]:
        sc += 10
        why.append("زخم MACD")
    if 45 <= rs <= 65:
        sc += 15
        why.append("RSI صحي")
    elif rs > 75:
        sc -= 10
        why.append("تشبع شراء")
    if ret > bench:
        sc += 15
        why.append("أقوى من السوق")
    if vr > 1.2 and c.iloc[-1] > c.iloc[-2]:
        sc += 5
        why.append("حجم داعم")
    if dist > -0.10:
        sc += 10
        why.append("قرب القمة")
    stop = px - 2 * A
    tgt = px + 2 * (px - stop)
    if stop <= 0:
        return None
    if sc >= 70 and ext <= 2.5 and rs < 70:
        rec = "ENTER"
    elif sc >= 55:
        rec = "PULLBACK" if ext > 2.5 or rs >= 70 else "WATCH"
    else:
        rec = "AVOID"
    return {
        "s": sym, "p": round(px, 2), "score": sc, "rec": rec,
        "rsi": round(rs), "ret6m": round(float(ret) * 100, 1),
        "stop": round(stop, 2), "tgt": round(tgt, 2), "why": why,
    }


def finite_number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def fundamentals(sym):
    import yfinance as yf
    t = yf.Ticker(sym)
    try:
        info = t.info or {}
    except Exception as exc:
        print(f"{sym}: fundamentals unavailable: {exc}", flush=True)
        info = {}
    fpe = finite_number(info.get("forwardPE"))
    peg = finite_number(info.get("pegRatio") or info.get("trailingPegRatio"))
    rg = finite_number(info.get("revenueGrowth"))
    eg = finite_number(info.get("earningsGrowth"))
    pm = finite_number(info.get("profitMargins"))
    days = None
    try:
        ed = (t.calendar or {}).get("Earnings Date")
        if ed:
            d = ed[0] if isinstance(ed, (list, tuple)) else ed
            d = d.date() if isinstance(d, datetime) else d
            if isinstance(d, date):
                days = (d - datetime.now(timezone.utc).date()).days
    except Exception as exc:
        print(f"{sym}: earnings calendar unavailable: {exc}", flush=True)
    if all(v is None for v in (fpe, rg, eg, pm)):
        return None, [], days
    f, notes = 0, []
    if rg is not None and rg > 0.10:
        f += 5
        notes.append(f"نمو إيراد {rg:.0%}")
    if eg is not None and eg > 0.10:
        f += 5
        notes.append(f"نمو أرباح {eg:.0%}")
    if pm is not None and pm > 0.15:
        f += 5
        notes.append(f"هامش {pm:.0%}")
    if fpe is not None:
        if 0 < fpe < 30:
            f += 5
            notes.append(f"P/E {fpe:.0f}")
        elif fpe > 60:
            f -= 5
            notes.append(f"P/E مرتفع {fpe:.0f}")
    if peg is not None and 0 < peg < 2:
        f += 5
        notes.append(f"PEG {peg:.1f}")
    return max(0, f), notes, days


def main():
    spy = get("SPY")
    if len(spy) < 126:
        raise RuntimeError("Insufficient SPY history; existing data.json kept")
    bench = float(spy["Close"].iloc[-1] / spy["Close"].iloc[-126] - 1)
    rows, skipped = [], []
    for s in SYMBOLS:
        try:
            row = analyze(s, get(s), bench)
            if row is None:
                raise RuntimeError("Insufficient or invalid price data")
            f, notes, days = fundamentals(s)
            row["total"] = round(row["score"] if f is None else 0.7 * row["score"] + 0.3 * f * 4)
            row["earn_days"] = days
            if row["rec"] == "ENTER":
                if row["total"] < 65:
                    row["rec"] = "WATCH"
                elif days is not None and 0 <= days <= 7:
                    row["rec"] = "EARNINGS"
            row["why"] += notes
            # Strict JSON avoids publishing NaN or Infinity to the browser.
            json.dumps(row, allow_nan=False)
            rows.append(row)
            print(f"{s}: OK", flush=True)
        except Exception as exc:
            skipped.append(s)
            print(f"{s}: skipped: {exc}", flush=True)
    if not rows:
        raise RuntimeError("No stocks updated; existing data.json kept")
    rows.sort(key=lambda row: row["total"], reverse=True)
    out = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "rows": rows,
        "skipped": skipped,
    }
    # Validate the complete payload before replacing the last good snapshot.
    payload = json.dumps(out, ensure_ascii=False, allow_nan=False, indent=2)
    output = Path(__file__).resolve().with_name("data.json")
    temp = output.with_suffix(".json.tmp")
    temp.write_text(payload + "\n", encoding="utf-8")
    temp.replace(output)
    print(f"Updated {len(rows)} stocks; skipped {len(skipped)}", flush=True)


if __name__ == "__main__":
    main()
