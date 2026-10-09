"""
NHL PLUS — projection model (pure Python, no Streamlit, unit-testable).

Design
------
* Team and player rates are regressed toward a prior: last season's rate
  (itself pulled 30% toward league average), weighted as K "phantom" games.
  Early in the season the prior dominates; by mid-season the current
  season does.
* Team goals = expected shots on goal x team shooting % x opposing starter's
  save-quality factor, adjusted for home ice, back-to-backs and recent form.
* Win probability: Poisson goals grid; ties go to OT/SO split by strength.
* Shots and saves use a negative binomial (overdispersed) distribution, because
  their real game-to-game spread is wider than Poisson assumes.
* Moneyline edge compares model probability with the book's NO-VIG probability.
"""

from __future__ import annotations

import math
from collections import defaultdict

# ------------------------------------------------------------------ constants
LEAGUE = {"gf": 3.05, "sog": 28.5, "sv": 0.900}   # per team per game; goalie SV%
POS_PRIOR = {  # per-game priors for skaters with little NHL history
    "F": {"sog": 1.9, "g": 0.22},
    "D": {"sog": 1.2, "g": 0.06},
}
TEAM_K = 15          # phantom games of prior weight for team rates
SKATER_SOG_K = 10    # ... for skater shots
SKATER_G_K = 25      # ... for skater goals (noisier, so heavier prior)
GOALIE_SV_K = 800    # phantom shots of prior weight for goalie save %
PRIOR_REGRESS = 0.30 # pull last season 30% toward league/position average
HOME_ADV = 1.03      # home scoring multiplier (away gets 2 - HOME_ADV)
B2B_PENALTY = 0.04   # tired team scores 4% less and allows 4% more
FORM_WEIGHT = 0.20   # weight on last-10 vs season rate
FORM_CAP = 0.10      # form can move a rate at most +/-10%
SOG_DISPERSION = 8.0     # negative binomial r for skater shots
SAVES_DISPERSION = 30.0  # negative binomial r for goalie saves


# ------------------------------------------------------------------ distributions
def pois(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def nb_pmf(k: int, mean: float, r: float) -> float:
    """Negative binomial with mean `mean` and variance mean + mean^2 / r."""
    if mean <= 0:
        return 1.0 if k == 0 else 0.0
    p = r / (r + mean)
    return math.exp(
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        + r * math.log(p) + k * math.log(1 - p)
    )


def p_at_least(k: int, mean: float, r: float | None = None) -> float:
    """P(X >= k). Poisson when r is None, otherwise negative binomial."""
    if k <= 0:
        return 1.0
    pmf = (lambda i: nb_pmf(i, mean, r)) if r else (lambda i: pois(i, mean))
    return min(1.0, max(0.0, 1.0 - sum(pmf(i) for i in range(k))))


def win_prob(lam_h: float, lam_a: float, max_goals: int = 15) -> float:
    ph = pa = pt = 0.0
    for i in range(max_goals):
        pi = pois(i, lam_h)
        for j in range(max_goals):
            p = pi * pois(j, lam_a)
            if i > j:
                ph += p
            elif i < j:
                pa += p
            else:
                pt += p
    # normalise away the mass lost by truncating the grid
    return (ph + pt * lam_h / (lam_h + lam_a)) / (ph + pa + pt)


# ------------------------------------------------------------------ odds
def to_american(p: float) -> int:
    p = min(max(p, 0.01), 0.99)
    return -round(100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def fmt_odds(o) -> str:
    if o is None:
        return "—"
    return f"+{o}" if o > 0 else str(o)


def implied(o: float) -> float:
    o = float(o)
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)


def no_vig(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Strip the bookmaker margin from a two-way market."""
    ia, ib = implied(odds_a), implied(odds_b)
    s = ia + ib
    return ia / s, ib / s


# ------------------------------------------------------------------ shrinkage
def prior_rate(prev_total: float, prev_gp: float, base: float,
               regress: float = PRIOR_REGRESS, min_gp: int = 10) -> float:
    """Last season's per-game rate, pulled toward `base`. Falls back to base."""
    if prev_gp >= min_gp:
        return (1 - regress) * prev_total / prev_gp + regress * base
    return base


def blend_rate(cur_total: float, cur_gp: float, prior: float, k: float) -> float:
    """Current-season rate regressed toward `prior` with weight k games."""
    return (cur_total + k * prior) / (cur_gp + k)


def goalie_sv(saves: float, shots: float, k: float = GOALIE_SV_K,
              league_sv: float = LEAGUE["sv"]) -> float:
    return (saves + k * league_sv) / (shots + k)


def goalie_factor(sv: float, league_sv: float = LEAGUE["sv"]) -> float:
    """>1 means a weaker-than-average goalie (more goals allowed)."""
    return (1 - sv) / (1 - league_sv)


def form_factor(recent_rate: float, season_rate: float, n_recent: int,
                weight: float = FORM_WEIGHT, cap: float = FORM_CAP) -> float:
    if n_recent < 5 or season_rate <= 0:
        return 1.0
    f = 1 + weight * (recent_rate / season_rate - 1)
    return min(max(f, 1 - cap), 1 + cap)


# ------------------------------------------------------------------ game model
def project_game(home: dict, away: dict, home_sv: float, away_sv: float,
                 league: dict = LEAGUE) -> dict:
    """
    home/away dicts need: sog, sa, sh (shooting %), off_form, def_form, b2b (bool).
    home_sv / away_sv: projected starting goalie save % for that team.
    """
    def side(me, opp, opp_sv, venue):
        x_sog = me["sog"] * opp["sa"] / league["sog"]
        x_g = x_sog * me["sh"] * goalie_factor(opp_sv, league["sv"])
        x_g *= venue * me["off_form"] * opp["def_form"]
        if me["b2b"]:
            x_g *= 1 - B2B_PENALTY
        if opp["b2b"]:
            x_g *= 1 + B2B_PENALTY
        return x_sog, x_g

    sog_h, lam_h = side(home, away, away_sv, HOME_ADV)
    sog_a, lam_a = side(away, home, home_sv, 2 - HOME_ADV)
    return {"sog_h": sog_h, "sog_a": sog_a, "lam_h": lam_h, "lam_a": lam_a,
            "ph": win_prob(lam_h, lam_a)}


# ------------------------------------------------------------------ backtest
def backtest(games: list[dict], k: float = TEAM_K, home_adv: float = HOME_ADV,
             league_gf: float = LEAGUE["gf"]) -> list[dict]:
    """
    Walk-forward moneyline backtest on goals only (no look-ahead):
    each game is predicted from results strictly before it.
    games: dicts with date, id, home, away, home_score, away_score.
    """
    stats = defaultdict(lambda: [0.0, 0.0, 0])  # gf, ga, gp
    out = []
    for g in sorted(games, key=lambda x: (x["date"], x["id"])):
        h, a = stats[g["home"]], stats[g["away"]]
        hgf, hga = blend_rate(h[0], h[2], league_gf, k), blend_rate(h[1], h[2], league_gf, k)
        agf, aga = blend_rate(a[0], a[2], league_gf, k), blend_rate(a[1], a[2], league_gf, k)
        lam_h = hgf * aga / league_gf * home_adv
        lam_a = agf * hga / league_gf * (2 - home_adv)
        p = win_prob(lam_h, lam_a)
        y = 1 if g["home_score"] > g["away_score"] else 0
        out.append({**g, "p_home": p, "home_win": y})
        h[0] += g["home_score"]; h[1] += g["away_score"]; h[2] += 1
        a[0] += g["away_score"]; a[1] += g["home_score"]; a[2] += 1
    return out


def score_predictions(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    eps = 1e-9
    acc = sum((r["p_home"] >= 0.5) == bool(r["home_win"]) for r in rows) / n
    brier = sum((r["p_home"] - r["home_win"]) ** 2 for r in rows) / n
    logloss = -sum(
        r["home_win"] * math.log(max(r["p_home"], eps))
        + (1 - r["home_win"]) * math.log(max(1 - r["p_home"], eps))
        for r in rows) / n
    return {"games": n, "accuracy": acc, "brier": brier, "log_loss": logloss,
            "brier_coinflip": 0.25, "log_loss_coinflip": math.log(2),
            "home_win_rate": sum(r["home_win"] for r in rows) / n}


def calibration(rows: list[dict], bins: int = 10) -> list[dict]:
    buckets = defaultdict(list)
    for r in rows:
        buckets[min(int(r["p_home"] * bins), bins - 1)].append(r)
    out = []
    for b in sorted(buckets):
        rs = buckets[b]
        out.append({"bin": f"{b * 100 // bins}–{(b + 1) * 100 // bins}%",
                    "predicted": sum(r["p_home"] for r in rs) / len(rs),
                    "actual": sum(r["home_win"] for r in rs) / len(rs),
                    "games": len(rs)})
    return out


# ------------------------------------------------------------------ pick grading
# A pick's edge is (model probability - break-even probability at the price), in points.
GRADE_SCALE = [("A+", 7.0), ("A", 5.0), ("B+", 3.5), ("B", 2.0), ("C", 0.5)]
GRADE_ORDER = {g: i for i, (g, _) in enumerate(GRADE_SCALE)}
ANYTIME_VIG = 1.08                        # books shade anytime-goal prices ~8%


def grade(edge_pts: float, penalties: int = 0) -> str | None:
    """Letter grade from edge in probability points; each penalty drops one step.
    Returns None when there's no meaningful edge (no pick)."""
    for i, (g, cut) in enumerate(GRADE_SCALE):
        if edge_pts >= cut:
            return GRADE_SCALE[min(i + penalties, len(GRADE_SCALE) - 1)][0]
    return None


TWO_WAY_VIG = 1.07   # a fair 50% prop hangs at about -115


def est_line(baseline_mean: float, r: float | None = None) -> float:
    """The half-point line a book would likely hang: where the player's baseline
    (pre-matchup) distribution is closest to a 50/50 over/under."""
    best_k = min(range(1, 80), key=lambda k: abs(p_at_least(k, baseline_mean, r) - 0.5))
    return best_k - 0.5


def prop_pick(baseline_mean: float, model_mean: float, r: float | None = None):
    """Pick a side of the estimated line. The book's price is approximated from the
    baseline probability plus vig, so edge comes only from what the matchup adds.
    Returns (side, line, model_prob, est_book_prob)."""
    line = est_line(baseline_mean, r)
    k = int(line + 0.5)
    p_base, p_mod = p_at_least(k, baseline_mean, r), p_at_least(k, model_mean, r)
    if p_mod >= p_base:
        side, p, pb = "Over", p_mod, p_base
    else:
        side, p, pb = "Under", 1 - p_mod, 1 - p_base
    return side, line, p, min(0.95, pb * TWO_WAY_VIG)


def worth_it_price(p_model: float, min_edge: float = 0.02) -> int:
    """Worst American price still worth betting: model edge >= min_edge (default = grade B)."""
    be = min(max(p_model - min_edge, 0.01), 0.99)
    return -math.floor(100 * be / (1 - be)) if be >= 0.5 else math.ceil(100 * (1 - be) / be)


def worth_it_label(o: int) -> str:
    """Reads naturally for both favourites and underdogs."""
    return f"{fmt_odds(o)} or better"


GRADE_CUTS = {g: cut for g, cut in GRADE_SCALE}


def grade_ladder(p_model: float) -> dict:
    """Worst price that still earns each grade, e.g. {'B': -108, 'A': 105, 'A+': 120}.
    None when the probability can't support that grade at any sane price."""
    out = {}
    for g in ("B", "A", "A+"):
        if p_model - GRADE_CUTS[g] / 100 <= 0.02:
            out[g] = None
        else:
            out[g] = worth_it_price(p_model, GRADE_CUTS[g] / 100)
    return out
