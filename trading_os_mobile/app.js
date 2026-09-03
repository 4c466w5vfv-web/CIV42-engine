const MAX_OPEN_RISK = 1.50;
const MAX_CLUSTER_RISK = 0.75;
const DEFAULT_THESIS_RISK = 0.75;

const state = {
  startEquity: 200000,
  equity: 204600,
  peakEquity: 205400,
  positions: [
    {symbol:'NVDA', side:'LONG', cluster:'Semiconductor', r:1.4, openRisk:0.00, protected:true},
    {symbol:'XOM', side:'LONG', cluster:'Energy', r:-0.2, openRisk:0.50, protected:false},
    {symbol:'TLT', side:'LONG', cluster:'Rates', r:0.8, openRisk:0.25, protected:false}
  ],
  signalIndex:0,
  signals:[
    {symbol:'AVGO', cluster:'Semiconductor', bigView:true, trend:true, zone:true, trigger:true, entry:356.20, stop:348.40},
    {symbol:'GC', cluster:'Metals', bigView:true, trend:true, zone:true, trigger:true, entry:3530.0, stop:3495.0},
    {symbol:'AMD', cluster:'Semiconductor', bigView:true, trend:true, zone:false, trigger:false, entry:210.0, stop:203.0},
    {symbol:'JPM', cluster:'Financials', bigView:true, trend:true, zone:true, trigger:true, entry:310.0, stop:304.0}
  ],
  journal:[
    {symbol:'NVDA', result:'+1.4R', note:'BE protected · runner active', good:true},
    {symbol:'XOM', result:'-0.2R', note:'initial risk active', good:false}
  ],
  violations:0
};

function money(n){return new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(n)}
function pct(n){return `${n.toFixed(2)}%`}
function openRisk(){return state.positions.reduce((s,p)=>s+p.openRisk,0)}
function availableRisk(){return Math.max(0,MAX_OPEN_RISK-openRisk())}
function drawdown(){return ((state.equity-state.peakEquity)/state.peakEquity)*100}
function buffer(){return ((state.equity-state.startEquity)/state.startEquity)*100}
function clusterRisk(name){return state.positions.filter(p=>p.cluster===name).reduce((s,p)=>s+p.openRisk,0)}
function clusters(){const out={};state.positions.forEach(p=>out[p.cluster]=(out[p.cluster]||0)+p.openRisk);return out}
function currentSignal(){return state.signals[state.signalIndex%state.signals.length]}
function riskAmount(){return state.equity*(DEFAULT_THESIS_RISK/100)}
function stopDistance(sig){return Math.abs(sig.entry-sig.stop)}
function unitSize(sig){const d=stopDistance(sig);return d>0?riskAmount()/d:0}
function signalDecision(sig){
  if(!sig.bigView||!sig.trend||!sig.zone||!sig.trigger) return {status:'WAIT', reason:'setup incomplete'};
  if(state.positions.some(p=>p.symbol===sig.symbol && p.openRisk>0)) return {status:'BLOCK', reason:'existing thesis active'};
  if(availableRisk()<DEFAULT_THESIS_RISK) return {status:'BLOCK', reason:'portfolio risk full'};
  if(clusterRisk(sig.cluster)+DEFAULT_THESIS_RISK>MAX_CLUSTER_RISK) return {status:'BLOCK', reason:'cluster risk full'};
  return {status:'ENTER', reason:'risk + setup approved'};
}
function setStateBadge(el,status){el.textContent=status;el.className='state '+status.toLowerCase()}
function render(){
  document.getElementById('clock').textContent=new Date().toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'});
  document.getElementById('equity').textContent=money(state.equity);
  document.getElementById('portfolioEquity').textContent=money(state.equity);
  document.getElementById('openRisk').textContent=pct(openRisk());
  document.getElementById('portfolioOpen').textContent=pct(openRisk());
  document.getElementById('availableRisk').textContent=pct(availableRisk());
  document.getElementById('drawdown').textContent=pct(drawdown());
  document.getElementById('buffer').textContent=pct(buffer());

  const sig=currentSignal();
  const decision=signalDecision(sig);
  setStateBadge(document.getElementById('actionState'),decision.status);
  setStateBadge(document.getElementById('tradeState'),decision.status);

  document.getElementById('positions').innerHTML=state.positions.map(p=>`<div class="card"><div class="row"><div><div class="ticker">${p.symbol} <span class="small">${p.side}</span></div><div class="small">${p.cluster} · ${p.protected?'BE protected':'risk active'}</div></div><div class="${p.r>=0?'rpos':'rneg'}">${p.r>=0?'+':''}${p.r.toFixed(1)}R</div></div><div class="small" style="margin-top:8px">Open risk ${pct(p.openRisk)}</div></div>`).join('');

  const cls=clusters();
  document.getElementById('riskMap').innerHTML=Object.entries(cls).map(([k,v])=>`<div class="card"><div class="row"><strong>${k}</strong><span>${pct(v)} / ${pct(MAX_CLUSTER_RISK)}</span></div><div class="bar"><div class="fill ${v>=MAX_CLUSTER_RISK?'bad':v>MAX_CLUSTER_RISK*.65?'warn':''}" style="width:${Math.min(100,v/MAX_CLUSTER_RISK*100)}%"></div></div></div>`).join('');

  document.getElementById('nextSignal').innerHTML=`<div class="card"><div class="signal"><div><div class="ticker">${sig.symbol}</div><div class="small">${sig.cluster}</div></div><strong>${decision.reason}</strong></div><div class="checks"><div class="chip ${sig.bigView?'ok':'no'}">BigView ${sig.bigView?'✓':'✕'}</div><div class="chip ${sig.trend?'ok':'no'}">HTF Trend ${sig.trend?'✓':'✕'}</div><div class="chip ${sig.zone?'ok':'no'}">HTF Zone ${sig.zone?'✓':'✕'}</div><div class="chip ${sig.trigger?'ok':'no'}">Trigger ${sig.trigger?'✓':'✕'}</div></div></div>`;

  document.getElementById('tradeTicker').textContent=`${sig.symbol} · ${sig.cluster}`;
  document.getElementById('tradeChecks').innerHTML=[['BigView',sig.bigView],['D1/H4 Trend',sig.trend],['H4/H1 Zone',sig.zone],['Trigger',sig.trigger],['Portfolio room',availableRisk()>=DEFAULT_THESIS_RISK],['Cluster room',clusterRisk(sig.cluster)+DEFAULT_THESIS_RISK<=MAX_CLUSTER_RISK]].map(([k,v])=>`<div class="chip ${v?'ok':'no'}">${k} ${v?'✓':'✕'}</div>`).join('');
  document.getElementById('tradeRisk').textContent=pct(DEFAULT_THESIS_RISK);
  document.getElementById('entryPrice').textContent=sig.entry.toLocaleString();
  document.getElementById('stopPrice').textContent=sig.stop.toLocaleString();
  document.getElementById('riskAmount').textContent=money(riskAmount());
  document.getElementById('unitSize').textContent=unitSize(sig).toLocaleString(undefined,{maximumFractionDigits:2});
  const btn=document.getElementById('executeTrade');btn.disabled=decision.status!=='ENTER';btn.textContent=decision.status==='ENTER'?'승인된 단일 진입 기록':decision.status;

  document.getElementById('clusterList').innerHTML=Object.entries(cls).map(([k,v])=>`<div class="card"><div class="row"><strong>${k}</strong><span>${pct(v)}</span></div><div class="bar"><div class="fill ${v>=MAX_CLUSTER_RISK?'bad':''}" style="width:${Math.min(100,v/MAX_CLUSTER_RISK*100)}%"></div></div></div>`).join('');
  document.getElementById('portfolioPositions').innerHTML=state.positions.map(p=>`<div class="card row"><div><strong>${p.symbol}</strong><div class="small">${p.cluster}</div></div><div style="text-align:right"><strong>${pct(p.openRisk)}</strong><div class="small">${p.protected?'protected':'at risk'}</div></div></div>`).join('');
  document.getElementById('journalCount').textContent=String(state.journal.length);
  document.getElementById('compliance').textContent=`${Math.max(0,100-state.violations*10)}%`;
  document.getElementById('journal').innerHTML=state.journal.slice().reverse().map(j=>`<div class="journal-item ${j.good?'good':'bad'}"><div class="row"><strong>${j.symbol}</strong><strong>${j.result}</strong></div><div class="small">${j.note}</div></div>`).join('')||'<div class="small">아직 기록 없음</div>';
}

document.querySelectorAll('.tab').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));btn.classList.add('active');document.querySelectorAll('.screen').forEach(x=>x.classList.toggle('active',x.dataset.screen===btn.dataset.tab))}));
document.getElementById('simulateNext').addEventListener('click',()=>{state.signalIndex++;render()});
document.getElementById('executeTrade').addEventListener('click',()=>{
  const sig=currentSignal(),decision=signalDecision(sig);if(decision.status!=='ENTER')return;
  state.positions.push({symbol:sig.symbol,side:'LONG',cluster:sig.cluster,r:0,openRisk:DEFAULT_THESIS_RISK,protected:false});
  state.journal.push({symbol:sig.symbol,result:'ENTRY',note:`single entry · risk ${pct(DEFAULT_THESIS_RISK)} · entry ${sig.entry} · stop ${sig.stop}`,good:true});
  render();
});
document.getElementById('addDemoExit').addEventListener('click',()=>{const p=state.positions.find(x=>x.openRisk>0);if(!p)return;p.openRisk=0;p.protected=true;p.r=2.8;state.equity+=2100;state.peakEquity=Math.max(state.peakEquity,state.equity);state.journal.push({symbol:p.symbol,result:'+2.8R',note:'20EMA partial → 50SMA final · risk recycled',good:true});render()});

render();setInterval(()=>document.getElementById('clock').textContent=new Date().toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'}),30000);
