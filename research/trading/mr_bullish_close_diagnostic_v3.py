from pathlib import Path
import pandas as pd

# Reuse the completed v2 diagnostic machinery, but stop before its report.
v2_path = Path(__file__).with_name('mr_absorption_exit_diagnostic_v2.py')
src = v2_path.read_text(encoding='utf-8')
marker = "print('# ARK-42 MR Absorption + Exit Diagnostic v2')"
cut = src.find(marker)
if cut < 0:
    raise RuntimeError('v2 report boundary not found')
ns = {'__name__':'mr_bullish_close_v3','__file__':str(v2_path)}
exec(compile(src[:cut], str(v2_path), 'exec'), ns)

rows = ns['rows']
SOURCES = ns['SOURCES']
core = ns['core']
fmt = ns['fmt']

# Add a no-lookahead H1 confirmation feature.
# v2 enters at accept_i OPEN, so the only candle-close confirmation known at entry
# is the immediately PREVIOUS completed H1 bar.
# bullish_close: Close > Open
# strong_bullish_close: bullish_close AND close-location-value >= 0.60
# where CLV=(Close-Low)/(High-Low).

prepared_h1 = {}
for sym, srcs in SOURCES.items():
    d1 = core['load'](srcs['D1'])
    h4 = core['load'](srcs['H4'])
    h1 = core['load'](srcs['H1'])
    d1, h4, h1, _ = core['prep'](d1, h4, h1)
    prepared_h1[sym] = h1

for r in rows:
    h1 = prepared_h1[r['symbol']]
    dt = pd.Timestamp(r['entry_dt'])
    i = int(h1['dt'].searchsorted(dt, side='left'))
    if i <= 0 or i >= len(h1) or pd.Timestamp(h1.iloc[i]['dt']) != dt:
        r['bullish_close'] = False
        r['strong_bullish_close'] = False
        r['confirm_clv'] = float('nan')
        continue
    b = h1.iloc[i-1]
    op = float(b['open']); hi = float(b['high']); lo = float(b['low']); cl = float(b['close'])
    bullish = cl > op
    clv = (cl-lo)/(hi-lo) if hi > lo else 0.5
    r['bullish_close'] = bool(bullish)
    r['confirm_clv'] = float(clv)
    r['strong_bullish_close'] = bool(bullish and clv >= 0.60)


def subset(fn):
    return [r for r in rows if fn(r)]

print('# ARK-42 MR Bullish-Close Confirmation Diagnostic v3')
print('IN-SAMPLE DIAGNOSTIC ONLY. This is a fixed-condition comparison on the same 149-event research set; not OOS validation.')
print('No-lookahead: bullish confirmation is the fully completed H1 candle immediately BEFORE the next-bar-open entry.')
print('Important-zone proximity is inherited from the existing shock builder. Exit A remains +1R 50% partial + 2ATR runner, 0.05R cost.')
print('Strong bullish close = Close>Open and Close Location Value >=0.60.')

base = subset(lambda r: r['decelerating'] and r['vol_contract'])
bull = subset(lambda r: r['decelerating'] and r['vol_contract'] and r['bullish_close'])
strong = subset(lambda r: r['decelerating'] and r['vol_contract'] and r['strong_bullish_close'])

print('\n## PRIMARY TEST: IMPORTANT ZONE + SMA200 DECELERATION + VOLUME CONTRACTION')
fmt('Base: deceleration + volume contraction', base, 'r_A')
fmt('+ bullish H1 close', bull, 'r_A')
fmt('+ strong bullish H1 close', strong, 'r_A')

full = subset(lambda r: r['decelerating'] and r['vol_contract'] and r['failed_downside'])
full_bull = subset(lambda r: r['decelerating'] and r['vol_contract'] and r['failed_downside'] and r['bullish_close'])
full_strong = subset(lambda r: r['decelerating'] and r['vol_contract'] and r['failed_downside'] and r['strong_bullish_close'])

print('\n## SECONDARY TEST: ADD TIGHT FAILED-DOWNSIDE FILTER')
fmt('Full absorption base', full, 'r_A')
fmt('Full + bullish H1 close', full_bull, 'r_A')
fmt('Full + strong bullish H1 close', full_strong, 'r_A')

print('\n## EXIT C CHECK: 50% RETRACE PARTIAL + 2ATR RUNNER')
fmt('Primary base C', base, 'r_C')
fmt('Primary + bullish C', bull, 'r_C')
fmt('Primary + strong bullish C', strong, 'r_C')

print('\n## ASSET/YEAR CONCENTRATION — PRIMARY STRONG-BULLISH')
if strong:
    df = pd.DataFrame(strong)
    print('n=', len(df))
    print('by_asset=', df.groupby('symbol')['r_A'].agg(['count','sum','mean']).round(3).to_dict('index'))
    print('by_year=', df.groupby('year')['r_A'].agg(['count','sum','mean']).round(3).to_dict('index'))
else:
    print('n=0')
