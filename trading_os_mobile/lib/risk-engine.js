export const RULES = {
  thesisRiskPct: 0.75,
  maxOpenRiskPct: 1.50,
  maxClusterRiskPct: 0.75,
  entryFractions: [0.40, 0.30, 0.30],
};

export function sumOpenRisk(positions = []) {
  return positions.reduce((sum, p) => sum + Number(p.openRisk || 0), 0);
}

export function clusterRisk(positions = [], cluster) {
  return positions
    .filter((p) => p.cluster === cluster)
    .reduce((sum, p) => sum + Number(p.openRisk || 0), 0);
}

export function evaluateSignal({ signal, positions = [], rules = RULES }) {
  if (!signal) return { status: 'WAIT', reason: 'no signal' };

  const requiredChecks = [
    ['bigView', 'BigView'],
    ['trend', 'HTF trend'],
    ['zone', 'HTF zone'],
    ['trigger', 'trigger'],
  ];
  const missing = requiredChecks.filter(([key]) => !signal[key]).map(([, label]) => label);
  if (missing.length) {
    return { status: 'WAIT', reason: `setup incomplete: ${missing.join(', ')}` };
  }

  const openRisk = sumOpenRisk(positions);
  const available = Math.max(0, rules.maxOpenRiskPct - openRisk);
  const cluster = clusterRisk(positions, signal.cluster);

  if (available + 1e-9 < rules.thesisRiskPct) {
    return { status: 'BLOCK', reason: 'portfolio risk full', openRisk, availableRisk: available, clusterRisk: cluster };
  }
  if (cluster + rules.thesisRiskPct > rules.maxClusterRiskPct + 1e-9) {
    return { status: 'BLOCK', reason: 'cluster risk full', openRisk, availableRisk: available, clusterRisk: cluster };
  }

  return { status: 'ENTER', reason: 'setup + risk approved', openRisk, availableRisk: available, clusterRisk: cluster };
}

export function calculatePositionSize({ equity, riskPct, entry, stop, pointValue = 1 }) {
  const e = Number(equity);
  const r = Number(riskPct);
  const en = Number(entry);
  const st = Number(stop);
  const pv = Number(pointValue);
  if (![e, r, en, st, pv].every(Number.isFinite) || e <= 0 || r <= 0 || pv <= 0) {
    throw new Error('invalid sizing input');
  }
  const stopDistance = Math.abs(en - st);
  if (stopDistance <= 0) throw new Error('entry and stop cannot match');
  const riskDollars = e * (r / 100);
  const units = riskDollars / (stopDistance * pv);
  return { riskDollars, stopDistance, units };
}

export function splitPlan(totalUnits, rules = RULES) {
  return rules.entryFractions.map((fraction, index) => ({
    slot: index + 1,
    fraction,
    units: Number(totalUnits) * fraction,
    riskPct: rules.thesisRiskPct * fraction,
  }));
}
