import io
import urllib.request
import pandas as pd
import research.trading.full_cycle_pilot as m


def fixed_load(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read().decode('utf-8-sig')
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = 'time' if 'time' in df.columns else ('datetime' if 'datetime' in df.columns else df.columns[0])
    # Some source rows span DST/offset changes. Normalize to naive UTC so .dt and asof ordering are stable.
    parsed = pd.to_datetime(df[tcol], errors='coerce', utc=True)
    df['dt'] = parsed.dt.tz_convert(None)
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    vcol = 'tick_volume' if 'tick_volume' in df.columns else ('volume' if 'volume' in df.columns else None)
    df['vol'] = pd.to_numeric(df[vcol], errors='coerce') if vcol else 0.0
    return df.dropna(subset=['dt','open','high','low','close']).sort_values('dt').drop_duplicates('dt').reset_index(drop=True)

m.load = fixed_load
m.main()
