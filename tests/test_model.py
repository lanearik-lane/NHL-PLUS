import math
import random

import pytest

import model as M


def test_distributions_sum_to_one():
    assert sum(M.pois(k, 3.1) for k in range(40)) == pytest.approx(1, abs=1e-9)
    assert sum(M.nb_pmf(k, 26.0, 30) for k in range(200)) == pytest.approx(1, abs=1e-9)


def test_negative_binomial_mean_and_variance():
    mean, r = 26.0, 30.0
    ks = range(300)
    mu = sum(k * M.nb_pmf(k, mean, r) for k in ks)
    var = sum((k - mu) ** 2 * M.nb_pmf(k, mean, r) for k in ks)
    assert mu == pytest.approx(mean, rel=1e-6)
    assert var == pytest.approx(mean + mean ** 2 / r, rel=1e-6)


def test_nb_less_extreme_than_poisson():
    # overdispersion pulls tail probabilities away from 0 and 1
    assert M.p_at_least(23, 34, M.SAVES_DISPERSION) < M.p_at_least(23, 34)
    assert M.p_at_least(45, 34, M.SAVES_DISPERSION) > M.p_at_least(45, 34)


def test_win_prob_symmetry_and_monotonic():
    assert M.win_prob(3, 3) == pytest.approx(0.5, abs=1e-9)
    assert M.win_prob(3.5, 3) > 0.5 > M.win_prob(3, 3.5)


def test_odds_roundtrip_and_no_vig():
    for p in (0.3, 0.5, 0.62):
        assert M.implied(M.to_american(p)) == pytest.approx(p, abs=0.005)
    h, a = M.no_vig(-150, 130)
    assert h + a == pytest.approx(1)
    assert h < M.implied(-150)  # margin removed


def test_shrinkage():
    prior = M.prior_rate(prev_total=250, prev_gp=82, base=1.9)
    assert 1.9 < prior < 250 / 82
    assert M.prior_rate(5, 3, 1.9) == 1.9           # too few games -> base
    assert M.blend_rate(0, 0, 2.0, 10) == 2.0         # no games -> prior
    assert M.blend_rate(40, 10, 2.0, 10) == pytest.approx(3.0)


def test_goalie_regression():
    hot = M.goalie_sv(190, 200)           # .950 over 200 shots
    assert M.LEAGUE["sv"] < hot < 0.95
    assert M.goalie_factor(0.92) < 1 < M.goalie_factor(0.88)


def test_form_capped():
    assert M.form_factor(10, 3, 10) == pytest.approx(1 + M.FORM_CAP)
    assert M.form_factor(3, 3, 3) == 1.0


def team(**kw):
    t = {"sog": 28.5, "sa": 28.5, "sh": 3.05 / 28.5, "off_form": 1, "def_form": 1, "b2b": False}
    t.update(kw)
    return t


def test_project_game_effects():
    base = M.project_game(team(), team(), 0.9, 0.9)
    assert base["ph"] > 0.5                                       # home ice
    assert base["lam_h"] == pytest.approx(3.05 * M.HOME_ADV)
    worse_goalie = M.project_game(team(), team(), 0.9, 0.88)      # away goalie worse
    assert worse_goalie["ph"] > base["ph"]
    tired = M.project_game(team(b2b=True), team(), 0.9, 0.9)
    assert tired["ph"] < base["ph"]


def test_backtest_no_lookahead_and_metrics():
    random.seed(1)
    teams = [f"T{i}" for i in range(8)]
    strength = {t: 2.4 + 0.2 * i for i, t in enumerate(teams)}
    games = []
    for n in range(600):
        h, a = random.sample(teams, 2)
        hs = sum(random.random() < strength[h] / 30 for _ in range(30))
        as_ = sum(random.random() < strength[a] / 30 for _ in range(30))
        if hs == as_:
            hs += 1
        games.append({"id": n, "date": f"2025-{10 + n // 200:02d}-{1 + n % 28:02d}",
                      "home": h, "away": a, "home_score": hs, "away_score": as_})
    rows = M.backtest(games)
    assert rows[0]["p_home"] == pytest.approx(M.win_prob(3.05 * M.HOME_ADV, 3.05 * (2 - M.HOME_ADV)))
    m = M.score_predictions(rows)
    assert m["games"] == 600
    assert m["brier"] < 0.25                         # learns team strength
    assert sum(c["games"] for c in M.calibration(rows)) == 600


def test_grades():
    assert M.grade(8) == "A+"
    assert M.grade(5.2) == "A"
    assert M.grade(2.1) == "B"
    assert M.grade(0.2) is None
    assert M.grade(8, penalties=1) == "A"
    assert M.grade(1, penalties=3) == "C"            # never falls below C


def test_est_line_and_prop_pick():
    r = M.SOG_DISPERSION
    assert M.est_line(1.4, r) == 1.5          # not a giveaway 0.5 line
    assert M.est_line(3.0, r) == 2.5
    assert 24.5 <= M.est_line(26.0, M.SAVES_DISPERSION) <= 26.5
    # no matchup change -> model equals book before vig -> negative edge, no pick
    side, line, p, pb = M.prop_pick(2.5, 2.5, r)
    assert p < pb and M.grade((p - pb) * 100) is None
    # big favourable matchup -> Over with positive edge
    side, line, p, pb = M.prop_pick(2.5, 3.4, r)
    assert side == "Over" and p > pb
    side, line, p, pb = M.prop_pick(2.5, 1.8, r)
    assert side == "Under" and p > pb
