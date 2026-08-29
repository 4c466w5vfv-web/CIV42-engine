"""v2.3 Effort-Result Inflection experiment.

Goal: test the user's core observable hypothesis BEFORE true CVD/footprint data is added:
large-volume candle -> poor continuation in that candle's direction -> opposite displacement -> retest -> LTF confirmation.

This is an OHLCV proxy experiment. It MUST NOT be described as true absorption, CVD, footprint,
or proof of institutional inventory. Those require exchange bid/ask trade data.

No OOS parameter selection. All thresholds are fixed ex ante from prior frozen research conventions.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import mtf_htf_zone_ltf_confirm_v16 as base
import mtf_htf_zone_ltf_confirm_v161 as causal
import mtf_exit_engine_v17 as exits
import weekly_daily_diagonal_v20 as v20

OUT = Path('research/results/inflection_effort_result_v23')
OUT.mkdir(parents=True, exist_ok=True)

VOL_SPIKE = v20.VOL_SPIKE          # 1.5x prior 20-bar volume baseline
ATR_N = v20.ATR_N                  # 14
FAIL_WINDOW_DAYS = 3
ORIGINAL_EXT_ATR_MAX = 0.25
OPPOSITE_DISP_ATR_MIN = 0.50
RETEST_DAYS = 10
RETEST_TOL_ATR = 0.20
LTF_LOOKBACK = 8
LTF_CONFIRM_HOURS = 24

STAGES = ('SPIKE_ONLY', 'FAILURE', 'DISPLACEMENT', 'RETEST_LTF')
EXIT_MODELS = ('FIXED_2R','FIXED_3R','FIXED_5R','PURE_SWING','ADAPTIVE_TRAIL')


def enrich(daily):
    return v20.enrich(daily)


def weekly_regime_series(daily):
    d = enrich(daily)
    w = v20.weekly_regime(daily)
    return w.reindex(d.index, method='ffill').fillna(0).astype(int)


def spike_events(daily):
    """Create causal event records from abnormal-volume daily candles.

    Candle direction is only a price-direction proxy. We do not infer aggressor side from OHLCV.
    A bearish spike creates a candidate LONG inflection; bullish spike creates SHORT.
    """
    d = enrich(daily)
    reg = weekly_regime_series(daily)
    rows = []
    for t, r in d.iterrows():
        if not np.isfinite(r.vol_ratio) or r.vol_ratio < VOL_SPIKE or not np.isfinite(r.atr):
            continue
        body = float(r.close - r.open)
        if body == 0:
            continue
        candle_dir = 1 if body > 0 else -1
        side = -candle_dir
        rows.append({
            'event_time': t,
            'signal_time': t,
            'side': side,
            'candle_dir': candle_dir,
            'open': float(r.open), 'high': float(r.high), 'low': float(r.low), 'close': float(r.close),
            'body_lo': float(min(r.open,r.close)), 'body_hi': float(max(r.open,r.close)),
            'mid': float((r.open+r.close)/2.0), 'atr': float(r.atr),
            'vol_ratio': float(r.vol_ratio), 'weekly_regime': int(reg.loc[t]),
            'stage': 'SPIKE_ONLY'
        })
    return pd.DataFrame(rows)


def classify_failure_and_displacement(events, daily):
    """Confirm failure only using bars AFTER event_time.

    LONG candidate (bearish spike): original-direction downside extension must stay <= .25 ATR,
    then a later close must reclaim the event midpoint (FAILURE), and a later/identical close must
    reclaim event open with >= .50 ATR opposite displacement (DISPLACEMENT).
    SHORT is symmetric.
    Signal time is the confirming daily bar, never event_time.
    """
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = enrich(daily)
    fail_rows, disp_rows = [], []
    for _, e in events.iterrows():
        et = pd.Timestamp(e.event_time)
        future = d[(d.index > et) & (d.index <= et + pd.Timedelta(days=FAIL_WINDOW_DAYS))]
        if future.empty:
            continue
        side = int(e.side); atr = float(e.atr)
        if side == 1:
            original_ext = max(0.0, float(e.low) - float(future.low.min()))
        else:
            original_ext = max(0.0, float(future.high.max()) - float(e.high))
        if original_ext > ORIGINAL_EXT_ATR_MAX * atr:
            continue

        fail_t = None
        disp_t = None
        for t, r in future.iterrows():
            if side == 1:
                fail_ok = float(r.close) > float(e.mid)
                disp_amt = float(r.close) - float(e.close)
                disp_ok = float(r.close) >= float(e.open) and disp_amt >= OPPOSITE_DISP_ATR_MIN * atr
            else:
                fail_ok = float(r.close) < float(e.mid)
                disp_amt = float(e.close) - float(r.close)
                disp_ok = float(r.close) <= float(e.open) and disp_amt >= OPPOSITE_DISP_ATR_MIN * atr
            if fail_t is None and fail_ok:
                fail_t = t
            if disp_t is None and disp_ok:
                disp_t = t
        if fail_t is not None:
            q = e.to_dict(); q.update({'signal_time':fail_t,'stage':'FAILURE','original_ext_atr':original_ext/atr}); fail_rows.append(q)
        if disp_t is not None:
            q = e.to_dict(); q.update({'signal_time':disp_t,'stage':'DISPLACEMENT','original_ext_atr':original_ext/atr}); disp_rows.append(q)
    return pd.DataFrame(fail_rows), pd.DataFrame(disp_rows)


def ltf_confirm_from_time(event, m15, start_time, require_retest=False):
    """HTF determines location; M15 only confirms directional structure.

    m15 timestamps are already shifted by the causal loader to availability time.
    Entry is the confirming M15 close at that availability timestamp. A future version should
    additionally test first-M1-open execution after this signal for stricter execution semantics.
    """
    x = m15.copy()
    x['prev_hi'] = x.high.shift(1).rolling(LTF_LOOKBACK, min_periods=LTF_LOOKBACK).max()
    x['prev_lo'] = x.low.shift(1).rolling(LTF_LOOKBACK, min_periods=LTF_LOOKBACK).min()
    st = pd.Timestamp(start_time)
    end = st + pd.Timedelta(hours=LTF_CONFIRM_HOURS)
    w = x[(x.index > st) & (x.index <= end)]
    zl, zh = float(event.body_lo), float(event.body_hi)
    tol = RETEST_TOL_ATR * float(event.atr)
    for t, r in w.iterrows():
        if not np.isfinite(r.prev_hi) or not np.isfinite(r.prev_lo):
            continue
        interacted = True
        if require_retest:
            interacted = (float(r.low) <= zh + tol and float(r.high) >= zl - tol)
        if int(event.side) == 1:
            confirm = interacted and float(r.close) > float(r.prev_hi) and float(r.close) > float(r.open)
            stop = float(min(r.prev_lo, zl))
        else:
            confirm = interacted and float(r.close) < float(r.prev_lo) and float(r.close) < float(r.open)
            stop = float(max(r.prev_hi, zh))
        if not confirm:
            continue
        entry = float(r.close)
        if (int(event.side)==1 and stop>=entry) or (int(event.side)==-1 and stop<=entry):
            continue
        q = event.to_dict()
        q.update({'entry_time':t,'signal_available_time':t,'entry':entry,'stop':stop,
                  'risk':abs(entry-stop),'stop_model':'LTF_STRUCTURE_PLUS_EVENT_BODY'})
        return q
    return None


def stage_candidates(events, m15, stage):
    if events.empty:
        return pd.DataFrame()
    rows=[]
    for _,e in events.iterrows():
        q=ltf_confirm_from_time(e,m15,e.signal_time,require_retest=False)
        if q is not None:
            q['stage']=stage; rows.append(q)
    return pd.DataFrame(rows)


def retest_candidates(displacements, daily, m15):
    """After displacement, wait for a later daily revisit of the event body zone that rejects/reclaims.
    Then require M15 directional structure confirmation strictly after that daily retest signal.
    """
    if displacements.empty:
        return pd.DataFrame()
    d = enrich(daily); rows=[]
    for _,e in displacements.iterrows():
        st=pd.Timestamp(e.signal_time); atr=float(e.atr); zl=float(e.body_lo); zh=float(e.body_hi); side=int(e.side)
        w=d[(d.index>st)&(d.index<=st+pd.Timedelta(days=RETEST_DAYS))]
        retest_t=None
        for t,r in w.iterrows():
            tol=RETEST_TOL_ATR*atr
            interacted=float(r.low)<=zh+tol and float(r.high)>=zl-tol
            if side==1:
                accepted=interacted and float(r.close)>float(e.mid)
            else:
                accepted=interacted and float(r.close)<float(e.mid)
            if accepted:
                retest_t=t; break
        if retest_t is None:
            continue
        q=ltf_confirm_from_time(e,m15,retest_t,require_retest=False)
        if q is not None:
            q['stage']='RETEST_LTF'; q['retest_time']=retest_t; rows.append(q)
    return pd.DataFrame(rows)


def run_model(c,m1,label):
    if label=='FIXED_2R': return exits.fixed_tp(c,m1,2.0)
    if label=='FIXED_3R': return exits.fixed_tp(c,m1,3.0)
    if label=='FIXED_5R': return exits.fixed_tp(c,m1,5.0)
    if label=='PURE_SWING': return exits.trailing(c,m1,'PURE_SWING')
    if label=='ADAPTIVE_TRAIL': return exits.trailing(c,m1,'ADAPTIVE_TRAIL')
    raise ValueError(label)


def main():
    name=base.os.environ['ASSET']
    m15,m1,daily,prov=causal.load_asset_causal(name)
    spikes=spike_events(daily)
    failure,disp=classify_failure_and_displacement(spikes,daily)
    candidates={
        'SPIKE_ONLY':stage_candidates(spikes,m15,'SPIKE_ONLY'),
        'FAILURE':stage_candidates(failure,m15,'FAILURE'),
        'DISPLACEMENT':stage_candidates(disp,m15,'DISPLACEMENT'),
        'RETEST_LTF':retest_candidates(disp,daily,m15)
    }
    out=OUT/name; out.mkdir(parents=True,exist_ok=True)
    spikes.to_csv(out/'events_spike.csv',index=False)
    failure.to_csv(out/'events_failure.csv',index=False)
    disp.to_csv(out/'events_displacement.csv',index=False)
    rows=[]
    for stage,c in candidates.items():
        c.to_csv(out/f'candidates_{stage}.csv',index=False)
        for label in EXIT_MODELS:
            tr=run_model(c,m1,label)
            tr.to_csv(out/f'trades_{stage}_{label}.csv',index=False)
            if tr.empty:
                rows.append({'asset':name,'stage':stage,'exit_model':label,'split':'ALL','trades':0})
                continue
            tr['split']=tr.entry_time.map(base.split)
            for sp,g in [('ALL',tr),*list(tr.groupby('split'))]:
                m=exits.metrics_ext(g); m.update({'asset':name,'stage':stage,'exit_model':label,'split':sp}); rows.append(m)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    meta={
        'version':'2.3','asset':name,'provenance':prov,
        'hypothesis':'abnormal volume + failed continuation + opposite displacement + later retest may identify inflection zones',
        'scientific_label':'OHLCV effort-result proxy; NOT true CVD/footprint/absorption',
        'stages':STAGES,'volume_spike':VOL_SPIKE,'failure_window_days':FAIL_WINDOW_DAYS,
        'original_extension_atr_max':ORIGINAL_EXT_ATR_MAX,'opposite_displacement_atr_min':OPPOSITE_DISP_ATR_MIN,
        'retest_days':RETEST_DAYS,'retest_tolerance_atr':RETEST_TOL_ATR,'ltf_lookback':LTF_LOOKBACK,
        'entry':'HTF event/confirmation then later causal M15 structure break','stop':'prior M15 structure plus event body invalidation',
        'selection_rule':'DEV/VAL only; OOS verification only. Never choose stage/exit from OOS.',
        'known_limits':['OHLCV cannot infer aggressor side','proxy volume may not equal centralized futures volume','M15 close execution semantics should later be checked against first M1 open']
    }
    (out/'meta.json').write_text(json.dumps(meta,indent=2,default=str))
    print(pd.DataFrame(rows).to_markdown(index=False))

if __name__=='__main__':
    main()
