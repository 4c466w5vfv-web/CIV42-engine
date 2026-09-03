import assert from 'node:assert/strict';
import { RULES, evaluateSignal, calculatePositionSize, splitPlan } from './lib/risk-engine.js';

const positions = [
  { symbol: 'XOM', cluster: 'Energy', openRisk: 0.50 },
  { symbol: 'TLT', cluster: 'Rates', openRisk: 0.25 },
  { symbol: 'NVDA', cluster: 'Semiconductor', openRisk: 0.00 },
];

const gold = { symbol: 'GC', cluster: 'Metals', bigView: true, trend: true, zone: true, trigger: true };
const avgo = { symbol: 'AVGO', cluster: 'Semiconductor', bigView: true, trend: true, zone: true, trigger: true };

assert.equal(evaluateSignal({ signal: gold, positions }).status, 'ENTER');
assert.equal(evaluateSignal({ signal: avgo, positions: [...positions, { symbol: 'AMD', cluster: 'Semiconductor', openRisk: 0.75 }] }).status, 'BLOCK');
assert.equal(evaluateSignal({ signal: { ...gold, zone: false }, positions }).status, 'WAIT');

const size = calculatePositionSize({ equity: 200000, riskPct: 0.75, entry: 100, stop: 98, pointValue: 1 });
assert.equal(size.riskDollars, 1500);
assert.equal(size.units, 750);
const plan = splitPlan(size.units, RULES);
assert.deepEqual(plan.map(x => Math.round(x.units)), [300, 225, 225]);
assert.ok(Math.abs(plan.reduce((s, x) => s + x.riskPct, 0) - 0.75) < 1e-9);

console.log('Trading OS risk-engine tests passed');
