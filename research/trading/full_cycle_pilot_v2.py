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
    parsed = pd.to_datetime(df[tcol].astype(str), errors='coerce', format='mixed', utc=True)
    # Convert through an explicit Series to guarantee datetime64[ns], not object.
    df['dt'] = pd.Series(parsed.array.tz_convert(None), index=df.index, dtype='datetime64[ns]')
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    vcol = 'tick_volume' if 'tick_volume' in df.columns else ('volume' if 'volume' in df.columns else None)
    df['vol'] = pd.to_numeric(df[vcol], errors='coerce') if vcol else 0.0
    out = df.dropna(subset=['dt','open','high','low','close']).sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
    if not pd.api.types.is_datetime64_ns_dtype(out['dt'].dtype):
        raise RuntimeError(f'bad datetime dtype {out["dt"].dtype} for {url}')
    return out


def fixed_prep(d1, h4, h1):
    # Defensive normalization before original prep uses .dt.hour.
    for x in (d1, h4, h1):
        x['dt'] = pd.to_datetime(x['dt'], errors='coerce').astype('datetime64[ns]')
    return original_prep(d1, h4, h1)

original_prep = m.prep
m.load = fixed_load
m.prep = fixed_prep
m.main()
