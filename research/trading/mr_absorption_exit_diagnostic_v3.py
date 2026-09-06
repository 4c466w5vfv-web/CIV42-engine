from pathlib import Path
import math
import pandas as pd

# Load v2 machinery and stop before its report prints.
p = Path(__file__).with_name('mr_absorption_exit_diagnostic_v2.py')
s = p.read_text(encoding='utf-8')
cut = s.find("print('# ARK-42 MR Absorption + Exit Diagnostic v2')")
if cut < 0:
    raise RuntimeError('v2 report boundary not found')
ns = {'__name__':'mr_absorption_v3','__file__':str(p)}
exec(compile(s[:cut], str(p), 'exec'), ns)

rows = ns['rows']
SOURCES = ns['SOURCES']
core = ns['core']

# Confirmation candle = last completed H1 candle immediately before entry.
# A: deceleration + volume contraction
# B: A + bullish close (close > open)
# C: A + strong bullish close (close > open and close location >= 0.60 of bar range)

h1_cache = {}
for sym, src in SOURCES.items():
    h1 = core['load'](src['H1'])
    h1_cache[sym] = h1

for r in rows:
    sym = r['symbol']
    h1 = h1_cache[sym]
    edt = pd.Timestamp(r['entry_dt'])
    j = h1['dt'].searchsorted(edt, side='left') - 1
    bullish = False
    strong = False
    if j >= 0:
        c = h1.iloc[int(j)]
        o = float(c['open']); h = float(c['high']); l = float(c['low']); cl = float(c['close'])
        bullish = cl > o
        rng = h - l
        close_loc = ((cl - l) / rng) if rng > 0 else float('nan')
        strong = bullish and pd.notna(close_loc) and close_loc >= 0.60
    r['bullish_close'] = bullish
    r['strong_bullish_close'] = strong


def stats(sel, key='r_A'):
    vals = pd.Series([float(x[key]) for x in sel if pd.notna(x.get(key))], dtype=float)
    if vals.empty:
        return None
    wins = vals[vals > 0]; losses = vals[vals < 0]
    gp = float(wins.sum()); gl = float(-losses.sum())
    eq = vals.cumsum(); dd = eq - eq.cummax()
    return {
        'n': int(len(vals)),
        'win': float((vals > 0).mean()*100),
        'meanR': float(vals.mean()),
        'medianR': float(vals.median()),
        'netR': float(vals.sum()),
        'PF': float(gp/gl) if gl > 0 else math.inf,
        'maxDD': float(dd.min())
    }


def out(label, sel):
    st = stats(sel)
    if not st:
        print(label, '| n=0'); return
    print(f"{label} | n={st['n']} win={st['win']:.1f}% meanR={st['meanR']:+.3f} medianR={st['medianR']:+.3f} netR={st['netR']:+.2f} PF={st['PF']:.2f} maxDD={st['maxDD']:.2f}R")

base = [r for r in rows if r['decelerating'] and r['vol_contract']]
bull = [r for r in base if r['bullish_close']]
strong = [r for r in base if r['strong_bullish_close']]

print('# ARK-42 MR Bullish-Close Confirmation Diagnostic v3')
print('IN-SAMPLE DIAGNOSTIC ONLY. Entry confirmation candle is strictly the completed H1 bar immediately before entry.')
print('Common exit: +1R 50% partial + 2ATR runner, same v2 costs/rules.')
print('A = trend deceleration + post-shock volume contraction')
print('B = A + bullish H1 close')
print('C = A + strong bullish H1 close (close location >=60% of candle range)')

out('A Deceleration + Volume contraction', base)
out('B + Bullish close', bull)
out('C + Strong bullish close', strong)

print('\n## BY ASSET')
for sym in SOURCES:
    print('\n', sym)
    out('A', [r for r in base if r['symbol']==sym])
    out('B', [r for r in bull if r['symbol']==sym])
    out('C', [r for r in strong if r['symbol']==sym])

print('\n## BY YEAR — STRONG BULLISH')
if strong:
    df = pd.DataFrame(strong)
    print(df.groupby('year')['r_A'].agg(['count','sum','mean']).round(3).to_string())
