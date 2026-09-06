import traceback
import research.trading.full_cycle_pilot_v2 as v2
import research.trading.full_cycle_pilot as m

for sym in ['XAUUSD']:
    print('DEBUG', sym)
    try:
        src=m.SOURCES[sym]
        d1=v2.fixed_load(src['D1'])
        h4=v2.fixed_load(src['H4'])
        h1=v2.fixed_load(src['H1'])
        print('dtypes before prep', d1['dt'].dtype, h4['dt'].dtype, h1['dt'].dtype)
        out=v2.fixed_prep(d1,h4,h1)
        print('prep ok')
        print(m.run_asset(sym,src))
    except Exception:
        traceback.print_exc()
        raise
