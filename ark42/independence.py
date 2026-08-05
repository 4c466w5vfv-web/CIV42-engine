"""Independence diagnostics — the product's central honest measurement.

The engine's premise was "several disciplines analyse independently". The
calls ARE independent (each discipline sees only the problem, never another
discipline's output). But independence of *procedure* is not independence of
*judgement*: the same model family, given the same problem, tends to produce
the same worldview under different labels.

This module measures that instead of assuming it:

  pairwise Pearson r between disciplines' score vectors (over the
  (option, criterion) cells they both filled)
        │
        ▼
  mean r  →  effective number of independent perspectives
             n_eff = D / (1 + (D-1) * mean_r)
        │
        ▼
  rho used by the Monte Carlo copula

Measured on the first real run (8 disciplines): mean r = 0.934,
n_eff = 1.06. All eight picked the same option. That is the honest state of
the art for this design, and the product reports it rather than hiding it.

Why n_eff = D / (1 + (D-1)·r): this is the standard design-effect / effective
sample size for equicorrelated observations. r = 0 → n_eff = D (truly
independent). r = 1 → n_eff = 1 (one perspective repeated D times).
"""
from __future__ import annotations

import itertools

import numpy as np

#: Below this many shared cells, a correlation estimate is too noisy to report
#: as a measurement. We say so instead of quoting a number we do not trust.
MIN_SHARED_CELLS = 6

#: The copula needs rho < 1 to stay numerically sane. A measured r above this
#: is reported verbatim but clamped for the simulation, and the clamp is
#: recorded so the reader can see we did it.
RHO_CAP = 0.95


def measure(tensor, providers=None) -> dict:
    """Independence diagnostics for one run's ScoreTensor.

    `providers`: optional list aligned to tensor.disciplines giving the
    provider_id that answered each discipline. When supplied, correlation is
    split into within-provider (same model — expected high) and
    cross-provider (different models — the real independence signal). Without
    it, every discipline is treated as one provider (today's single-model
    default), which is reported honestly as single_provider.

    Returns a dict that is safe to persist and to show a user. Never raises
    on degenerate input — it reports `reliable: False` with a reason instead,
    because a missing measurement must not take a paid run down.
    """
    mean, conf = tensor.mean, tensor.conf
    disciplines = list(tensor.disciplines)
    D = len(disciplines)
    if providers is None or len(providers) != D:
        providers = [None] * D
    prov = {disciplines[i]: providers[i] for i in range(D)}
    distinct_prov = sorted({p for p in providers if p})
    n_providers = len(distinct_prov) if distinct_prov else 1

    out: dict = {
        "n_disciplines": D,
        "n_providers": n_providers,
        "providers": distinct_prov,
        "single_provider": n_providers <= 1,
        "method": ("각 학문이 공통으로 채운 (옵션×기준) 셀에 대한 피어슨 상관. "
                   "유효 독립 관점 수 n_eff = D / (1 + (D-1)·평균상관). "
                   "독립성은 서로 다른 공급자(모델) 사이에서만 진짜다."),
        "min_shared_cells": MIN_SHARED_CELLS,
        "rho_cap": RHO_CAP,
    }

    if D < 2:
        out.update(reliable=False, reason="학문이 1개 이하 — 상관을 정의할 수 없음",
                   mean_r=None, n_effective=float(D), rho_used=0.0,
                   shared_cells=0, unanimous_top=None, pairs=[])
        return out

    # Cells every discipline filled — comparing only where all overlap keeps
    # the pairwise correlations on the same footing.
    present = conf > 0                                   # (O, C, D)
    common = present.all(axis=2).ravel()                 # (O*C,)
    n_shared = int(common.sum())
    vecs = {d: np.nan_to_num(mean[:, :, i]).ravel()[common]
            for i, d in enumerate(disciplines)}

    if n_shared < MIN_SHARED_CELLS:
        out.update(reliable=False,
                   reason=f"공통 셀 {n_shared}개 < {MIN_SHARED_CELLS}개 — 추정 불가",
                   mean_r=None, n_effective=None, rho_used=0.0,
                   shared_cells=n_shared, unanimous_top=None, pairs=[])
        return out

    pairs = []
    for a, b in itertools.combinations(disciplines, 2):
        x, y = vecs[a], vecs[b]
        # A discipline that gave every shared cell the same score has zero
        # variance; correlation is undefined, not zero. Skip that pair.
        if x.std() == 0 or y.std() == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(r):
            same = (prov[a] is not None and prov[a] == prov[b])
            pairs.append({"a": a, "b": b, "r": round(r, 4),
                          "same_provider": bool(same)})

    if not pairs:
        out.update(reliable=False,
                   reason="분산이 0인 학문 때문에 상관을 계산할 수 없음",
                   mean_r=None, n_effective=None, rho_used=0.0,
                   shared_cells=n_shared, unanimous_top=None, pairs=[])
        return out

    rs = np.array([p["r"] for p in pairs])
    mean_r = float(rs.mean())
    # Negative mean correlation means the disciplines genuinely diverge — a
    # GOOD outcome for this product. n_eff is then capped at D (you cannot
    # have more independent perspectives than disciplines) and the copula
    # runs at rho = 0.
    n_eff = D / (1.0 + (D - 1) * mean_r) if mean_r > 0 else float(D)
    n_eff = min(n_eff, float(D))
    rho_used = float(min(max(mean_r, 0.0), RHO_CAP))

    # Do they all pick the same option? The bluntest possible independence
    # check, and the one a non-technical reader understands immediately.
    per_disc_top = []
    for i in range(D):
        m = np.where(present[:, :, i], mean[:, :, i], np.nan)
        # An option this discipline skipped entirely is an all-NaN row;
        # np.nanmean would warn on it, so mask those rows out first.
        covered = ~np.all(np.isnan(m), axis=1)
        if not covered.any():
            continue
        om = np.full(m.shape[0], np.nan)
        om[covered] = np.nanmean(m[covered], axis=1)
        per_disc_top.append(tensor.options[int(np.nanargmax(om))])

    # Within- vs cross-provider correlation. Same-provider pairs are the same
    # voice and are EXPECTED to correlate; cross-provider pairs are the actual
    # independence signal. Only reported when >1 provider is present.
    within = [p["r"] for p in pairs if p["same_provider"]]
    cross = [p["r"] for p in pairs if not p["same_provider"]]
    within_mean = round(float(np.mean(within)), 4) if within else None
    cross_mean = round(float(np.mean(cross)), 4) if cross else None

    out.update(
        reliable=True,
        reason="",
        shared_cells=n_shared,
        mean_r=round(mean_r, 4),
        min_r=round(float(rs.min()), 4),
        max_r=round(float(rs.max()), 4),
        within_provider_mean_r=within_mean,
        cross_provider_mean_r=cross_mean,
        n_effective=round(n_eff, 3),
        rho_used=round(rho_used, 4),
        rho_was_capped=bool(mean_r > RHO_CAP),
        unanimous_top=bool(len(set(per_disc_top)) == 1) if per_disc_top else None,
        per_discipline_top=per_disc_top,
        pairs=sorted(pairs, key=lambda p: -p["r"]),
        verdict=verdict_ko(D, mean_r, n_eff, n_providers, cross_mean),
    )
    return out


def verdict_ko(D: int, mean_r: float, n_eff: float,
               n_providers: int = 1, cross_mean_r=None) -> str:
    """One plain-Korean sentence a non-technical reader can act on."""
    # Single provider: name the structural cause, not just the symptom.
    if n_providers <= 1 and mean_r > 0.8:
        return (f"{D}개 학문이 사실상 하나의 관점처럼 판단했습니다 (상관 {mean_r:.2f}). "
                f"유효 독립 관점 {n_eff:.1f}개. 원인은 명확합니다 — {D}개 학문이 모두 "
                f"같은 모델 하나에서 나왔습니다. 진짜 독립 검증은 서로 다른 모델을 "
                f"요구합니다. 확률은 이 사실을 반영해 낮춰 보고됩니다.")
    # Multi-provider: the cross-provider number is the headline.
    if n_providers > 1 and cross_mean_r is not None:
        if cross_mean_r <= 0.3:
            return (f"{n_providers}개 서로 다른 모델이 이 문제를 분석했고, 모델 간 "
                    f"판단 상관은 {cross_mean_r:.2f}로 낮습니다 — 진짜 독립 관점이 "
                    f"확보됐습니다. 유효 독립 관점 {n_eff:.1f}개.")
        if cross_mean_r <= 0.7:
            return (f"{n_providers}개 모델의 판단이 부분적으로 겹칩니다 "
                    f"(모델 간 상관 {cross_mean_r:.2f}). 유효 독립 관점 {n_eff:.1f}개 — "
                    f"단일 모델보다는 독립적입니다.")
        return (f"{n_providers}개 모델을 썼지만 모델 간에도 판단이 크게 겹칩니다 "
                f"(상관 {cross_mean_r:.2f}). 유효 독립 관점 {n_eff:.1f}개 — 모델을 "
                f"바꿔도 문제 자체가 답을 좁히거나, 두 모델이 비슷한 편향을 공유합니다.")
    if mean_r <= 0.2:
        return (f"{D}개 학문이 서로 다른 판단을 했습니다 (상관 {mean_r:.2f}). "
                f"유효 독립 관점 {n_eff:.1f}개 — 다관점 분석이 실제로 작동했습니다.")
    if mean_r <= 0.5:
        return (f"{D}개 학문의 판단이 부분적으로 겹칩니다 (상관 {mean_r:.2f}). "
                f"유효 독립 관점 {n_eff:.1f}개.")
    if mean_r <= 0.8:
        return (f"{D}개 학문의 판단이 상당히 겹칩니다 (상관 {mean_r:.2f}). "
                f"유효 독립 관점 {n_eff:.1f}개 — 겉보기 학문 수보다 훨씬 적습니다.")
    return (f"{D}개 학문이 사실상 하나의 관점처럼 판단했습니다 (상관 {mean_r:.2f}). "
            f"유효 독립 관점 {n_eff:.1f}개 — 이 분석은 '{D}개의 독립 견해'가 "
            f"아니라 '1개 견해를 {D}번 확인한 것'에 가깝습니다. "
            f"확률은 이 사실을 반영해 낮춰 보고됩니다.")
