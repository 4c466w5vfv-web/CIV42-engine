import { RULES, evaluateSignal, calculatePositionSize, splitPlan } from '../lib/risk-engine.js';

function json(res, status, body) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(body));
}

export default async function handler(req, res) {
  if (req.method !== 'POST') return json(res, 405, { error: 'POST only' });

  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body || '{}') : (req.body || {});
    const positions = Array.isArray(body.positions) ? body.positions : [];
    const signal = body.signal || null;
    const decision = evaluateSignal({ signal, positions, rules: RULES });

    let sizing = null;
    if (decision.status === 'ENTER' && body.account && signal?.entry != null && signal?.stop != null) {
      const size = calculatePositionSize({
        equity: body.account.equity,
        riskPct: RULES.thesisRiskPct,
        entry: signal.entry,
        stop: signal.stop,
        pointValue: signal.pointValue ?? 1,
      });
      sizing = { ...size, split: splitPlan(size.units, RULES) };
    }

    return json(res, 200, {
      version: 'trading-guard-v1',
      rules: RULES,
      decision,
      sizing,
      signal,
    });
  } catch (error) {
    return json(res, 400, { error: error?.message || 'invalid request' });
  }
}
