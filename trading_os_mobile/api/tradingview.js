import { RULES, evaluateSignal } from '../lib/risk-engine.js';

function json(res, status, body) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(body));
}

function normalizeAlert(body = {}) {
  const side = String(body.side || body.action || 'LONG').toUpperCase();
  return {
    symbol: String(body.symbol || body.ticker || '').toUpperCase(),
    side,
    cluster: String(body.cluster || body.sector || 'Unclassified'),
    bigView: Boolean(body.bigView ?? body.bigview ?? true),
    trend: Boolean(body.trend ?? true),
    zone: Boolean(body.zone ?? true),
    trigger: Boolean(body.trigger ?? true),
    entry: body.entry != null ? Number(body.entry) : null,
    stop: body.stop != null ? Number(body.stop) : null,
    pointValue: body.pointValue != null ? Number(body.pointValue) : 1,
    receivedAt: new Date().toISOString(),
  };
}

export default async function handler(req, res) {
  if (req.method !== 'POST') return json(res, 405, { error: 'POST only' });
  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body || '{}') : (req.body || {});
    const signal = normalizeAlert(body);
    if (!signal.symbol) return json(res, 400, { error: 'symbol required' });

    // Stateless webhook adapter: positions can be sent by the caller until a persistent account store is attached.
    const positions = Array.isArray(body.positions) ? body.positions : [];
    const decision = evaluateSignal({ signal, positions, rules: RULES });

    return json(res, 200, {
      accepted: true,
      signal,
      decision,
      next: decision.status === 'ENTER' ? 'send to execution adapter after account-state validation' : 'do not execute',
    });
  } catch (error) {
    return json(res, 400, { error: error?.message || 'invalid TradingView payload' });
  }
}
