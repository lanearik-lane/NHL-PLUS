"""
NHL PLUS — NHL betting-angle dashboard
Targets: Moneyline, Shots on Goal, Goalie Saves, Goals

Data:   free public NHL web API (no key) — https://api-web.nhle.com/v1
        moneylines from the NHL's partner-odds feed (DraftKings, US) when posted
Images: NHL team logos + player headshots from assets.nhle.com
Run:    pip install -r requirements.txt && streamlit run app.py
"""

import datetime as dt
import math
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import requests
import streamlit as st

import model as M

API = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "NHL-PLUS/2.0"}
FINAL_STATES = ("FINAL", "OFF")
ALL_TEAMS = ["ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL", "DET", "EDM",
             "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR", "OTT", "PHI", "PIT",
             "SJS", "SEA", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WSH", "WPG"]

st.set_page_config(page_title="NHL PLUS", page_icon="🏒", layout="wide")
st.markdown(
    """
    <style>
      .nhlplus-title {font-size: 2.6rem; font-weight: 800; margin-bottom: 0;}
      .nhlplus-title span {color: #e4002b;}
      .game-card {border: 1px solid #2a2f3a; border-radius: 14px; padding: 14px 10px;
                  margin-bottom: 12px; text-align: center; background: #161a22;}
      .pick-card {border: 1px solid #2a2f3a; border-radius: 14px; padding: 14px 16px;
                  margin-bottom: 12px; background: #161a22; display: flex; gap: 14px; align-items: center;}
      .pick-card img {width: 58px; height: 58px; border-radius: 50%; background: #0e1117; object-fit: cover;}
      .pick-main {flex: 1; min-width: 0;}
      .pick-title {font-size: 1.05rem; font-weight: 700;}
      .pick-why {font-size: .82rem; opacity: .75; margin-top: 3px;}
      .grade {font-size: 1.5rem; font-weight: 800; min-width: 58px; text-align: center;
              border-radius: 12px; padding: 6px 0;}
      .g-A {background: #16a34a26; color: #4ade80;}
      .g-B {background: #f59e0b26; color: #fbbf24;}
      .g-C {background: #94a3b826; color: #cbd5e1;}
      .game-card img {width: 72px; height: 72px;}
      .matchup {font-size: 1.15rem; font-weight: 700;}
      .meta {font-size: .9rem; opacity: .8; line-height: 1.5;}
      .ml {font-size: 1.05rem; font-weight: 700;}
      .pill {display:inline-block; padding:2px 10px; border-radius:999px; background:#e4002b22;
             font-size:.8rem; font-weight:600; margin: 2px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================= API layer
@st.cache_data(ttl=600, show_spinner=False)
def get_json(path: str):
    r = requests.get(f"{API}/{path}", headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def safe(path: str):
    try:
        return get_json(path)
    except Exception:
        return None


def txt(v, default=""):
    if isinstance(v, dict):
        return v.get("default", default)
    return v if v is not None else default


def season_ids(d: dt.date):
    start = d.year if d.month >= 9 else d.year - 1
    return f"{start}{start + 1}", f"{start - 1}{start}"


def logo_url(abbrev: str) -> str:
    return f"https://assets.nhle.com/logos/nhl/svg/{abbrev}_dark.svg"


@st.cache_data(ttl=600, show_spinner=False)
def games_on(date_str: str):
    data = get_json(f"schedule/{date_str}")
    for day in data.get("gameWeek", []):
        if day.get("date") == date_str:
            return day.get("games", [])
    return []


@st.cache_data(ttl=300, show_spinner=False)
def moneylines(date_str: str) -> dict:
    """{gameId: (home_ml, away_ml)} from the NHL partner-odds feed."""
    data = safe(f"partner-game/US/{date_str}") or safe("partner-game/US/now") or {}

    def ml(team):
        for o in team.get("odds", []) or []:
            if o.get("description") == "MONEY_LINE_2_WAY":
                try:
                    return int(float(o.get("value")))
                except (TypeError, ValueError):
                    return None
        return None

    return {g.get("gameId"): (ml(g.get("homeTeam", {})), ml(g.get("awayTeam", {})))
            for g in data.get("games", [])}


@st.cache_data(ttl=3600, show_spinner=False)
def boxscore(game_id: int):
    return safe(f"gamecenter/{game_id}/boxscore")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def season_totals(player_id: int, season: str) -> dict | None:
    """A player's NHL regular-season totals for one season, summed across teams."""
    data = safe(f"player/{player_id}/landing")
    if not data:
        return None
    rows = [r for r in data.get("seasonTotals", [])
            if str(r.get("season")) == season and r.get("gameTypeId") == 2
            and r.get("leagueAbbrev") == "NHL"]
    if not rows:
        return None
    keys = ("gamesPlayed", "goals", "shots", "gamesStarted", "shotsAgainst", "goalsAgainst")
    tot = {k: sum(r.get(k, 0) or 0 for r in rows) for k in keys}
    tot["saves"] = tot["shotsAgainst"] - tot["goalsAgainst"]
    return tot


def _gp(cs):
    return max((s.get("gamesPlayed", 0) for s in (cs or {}).get("skaters", [])), default=0)


def _roster_people(abbrev, cs_cur, cs_prev):
    """Current roster; falls back to club-stats players if the roster call fails."""
    rost = safe(f"roster/{abbrev}/current")
    people = []
    if rost:
        for group in rost.values():
            if isinstance(group, list):
                for p in group:
                    people.append({"id": p.get("id"), "first": txt(p.get("firstName")),
                                   "last": txt(p.get("lastName")), "pos": p.get("positionCode", "F"),
                                   "headshot": p.get("headshot")})
    if people:
        return people, True
    seen = {}
    for cs in (cs_cur, cs_prev):
        for s in (cs or {}).get("skaters", []):
            seen.setdefault(s["playerId"], {"id": s["playerId"], "first": txt(s.get("firstName")),
                                            "last": txt(s.get("lastName")),
                                            "pos": s.get("positionCode", "F"), "headshot": s.get("headshot")})
        for g in (cs or {}).get("goalies", []):
            seen.setdefault(g["playerId"], {"id": g["playerId"], "first": txt(g.get("firstName")),
                                            "last": txt(g.get("lastName")), "pos": "G",
                                            "headshot": g.get("headshot")})
    return list(seen.values()), False


@st.cache_data(ttl=1800, show_spinner=False)
def team_model(abbrev: str, cur: str, prev: str, date_str: str) -> dict:
    L = M.LEAGUE
    cs_cur = safe(f"club-stats/{abbrev}/{cur}/2") or {}
    cs_prev = safe(f"club-stats/{abbrev}/{prev}/2") or {}
    sched = (safe(f"club-schedule-season/{abbrev}/{cur}") or {}).get("games", [])

    done = sorted((g for g in sched if g.get("gameType") == 2 and g.get("gameState") in FINAL_STATES
                   and g.get("gameDate", "") < date_str), key=lambda g: g["gameDate"])
    yday = (dt.date.fromisoformat(date_str) - dt.timedelta(days=1)).isoformat()
    b2b = any(g.get("gameDate") == yday and g.get("gameType") in (2, 3) for g in sched)

    def gf_ga(g):
        home = g["homeTeam"]["abbrev"] == abbrev
        me, opp = (g["homeTeam"], g["awayTeam"]) if home else (g["awayTeam"], g["homeTeam"])
        return me.get("score", 0), opp.get("score", 0)

    results = [gf_ga(g) for g in done]
    gp = len(results)
    cs_gp, prev_gp = _gp(cs_cur), _gp(cs_prev)

    def sums(cs):
        sk, gl = cs.get("skaters", []), cs.get("goalies", [])
        return (sum(s.get("shots", 0) for s in sk), sum(g.get("shotsAgainst", 0) for g in gl),
                sum(s.get("goals", 0) for s in sk), sum(g.get("goalsAgainst", 0) for g in gl))

    c_sog, c_sa, _, _ = sums(cs_cur)
    p_sog, p_sa, p_gf, p_ga = sums(cs_prev)
    K = M.TEAM_K
    sog = M.blend_rate(c_sog, cs_gp, M.prior_rate(p_sog, prev_gp, L["sog"]), K)
    sa = M.blend_rate(c_sa, cs_gp, M.prior_rate(p_sa, prev_gp, L["sog"]), K)
    gf = M.blend_rate(sum(r[0] for r in results), gp, M.prior_rate(p_gf, prev_gp, L["gf"]), K)
    ga = M.blend_rate(sum(r[1] for r in results), gp, M.prior_rate(p_ga, prev_gp, L["gf"]), K)

    last10 = results[-10:]
    n10 = len(last10)
    off_form = M.form_factor(sum(r[0] for r in last10) / n10, gf, n10) if n10 else 1.0
    def_form = M.form_factor(sum(r[1] for r in last10) / n10, ga, n10) if n10 else 1.0

    # who dressed in the most recent regular-season game
    lineup_ids = None
    if done:
        bx = boxscore(done[-1]["id"]) or {}
        side = "homeTeam" if bx.get("homeTeam", {}).get("abbrev") == abbrev else "awayTeam"
        pbg = bx.get("playerByGameStats", {}).get(side, {})
        ids = {p["playerId"] for grp in ("forwards", "defense", "goalies") for p in pbg.get(grp, [])}
        lineup_ids = ids or None

    people, from_roster = _roster_people(abbrev, cs_cur, cs_prev)
    cur_sk = {s["playerId"]: s for s in cs_cur.get("skaters", [])}
    prev_sk = {s["playerId"]: s for s in cs_prev.get("skaters", [])}
    cur_gl = {g["playerId"]: g for g in cs_cur.get("goalies", [])}
    prev_gl = {g["playerId"]: g for g in cs_prev.get("goalies", [])}

    skaters, goalies = [], []
    for p in people:
        pid = p["id"]
        name = f"{p['first']} {p['last']}".strip()
        if p["pos"] == "G":
            c = cur_gl.get(pid, {})
            pv = prev_gl.get(pid) or season_totals(pid, prev) or {}
            saves = c.get("saves", 0) + pv.get("saves", 0)
            shots = c.get("shotsAgainst", 0) + pv.get("shotsAgainst", 0)
            goalies.append({"id": pid, "name": name, "headshot": p["headshot"],
                            "sv": M.goalie_sv(saves, shots), "shots_sample": shots,
                            "starts_cur": c.get("gamesStarted", 0),
                            "starts_prev": pv.get("gamesStarted", 0),
                            "start_score": c.get("gamesStarted", 0) + 0.3 * pv.get("gamesStarted", 0)})
            continue
        pc = "D" if p["pos"] == "D" else "F"
        c = cur_sk.get(pid, {})
        pv = prev_sk.get(pid) or season_totals(pid, prev) or {}
        cgp, pgp = c.get("gamesPlayed", 0), pv.get("gamesPlayed", 0)
        prior_sog = M.prior_rate(pv.get("shots", 0), pgp, M.POS_PRIOR[pc]["sog"])
        prior_g = M.prior_rate(pv.get("goals", 0), pgp, M.POS_PRIOR[pc]["g"])
        skaters.append({
            "id": pid, "name": name, "pos": p["pos"], "headshot": p["headshot"],
            "gp": cgp, "prev_gp": pgp,
            "sog_rate": M.blend_rate(c.get("shots", 0), cgp, prior_sog, M.SKATER_SOG_K),
            "g_rate": M.blend_rate(c.get("goals", 0), cgp, prior_g, M.SKATER_G_K),
            "in_lineup": lineup_ids is None or pid in lineup_ids,
        })

    goalies.sort(key=lambda g: g["start_score"], reverse=True)
    return {"abbrev": abbrev, "gp": gp, "sog": sog, "sa": sa, "gf": gf, "ga": ga,
            "sh": gf / sog if sog else L["gf"] / L["sog"],
            "off_form": off_form, "def_form": def_form, "b2b": b2b,
            "lineup_known": lineup_ids is not None, "from_roster": from_roster,
            "skaters": skaters, "goalies": goalies}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def season_results(season: str) -> list[dict]:
    games = {}
    for t in ALL_TEAMS:
        for g in (safe(f"club-schedule-season/{t}/{season}") or {}).get("games", []):
            if g.get("gameType") == 2 and g.get("gameState") in FINAL_STATES:
                games[g["id"]] = {"id": g["id"], "date": g["gameDate"],
                                  "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                                  "home_score": g["homeTeam"].get("score", 0),
                                  "away_score": g["awayTeam"].get("score", 0)}
    return list(games.values())


# ================================================================= sidebar (part 1)
eastern_today = dt.datetime.now(ZoneInfo("America/New_York")).date()
with st.sidebar:
    st.markdown("## 🏒 NHL PLUS")
    pick_date = st.date_input("Game date", eastern_today)
    tz_name = st.selectbox("Show times in", ["America/New_York", "America/Chicago", "America/Denver",
                                             "America/Los_Angeles", "America/Phoenix", "UTC"])
    min_gp = st.slider("Min NHL games (this + last season)", 0, 60, 10)
    show_n = st.slider("Rows per table", 10, 100, 30)
    lineup_only = st.toggle("Only players who dressed last game", value=True,
                            help="Uses the team's most recent regular-season boxscore. "
                                 "Ignored until a team has played a game.")

st.markdown('<p class="nhlplus-title">NHL <span>PLUS</span></p>', unsafe_allow_html=True)
st.caption("Moneyline · Shots on Goal · Goalie Saves · Goals")

date_str = pick_date.isoformat()
cur_season, prev_season = season_ids(pick_date)
tz = ZoneInfo(tz_name)

try:
    games = games_on(date_str)
except Exception as e:
    st.error(f"Couldn't reach the NHL API: {e}")
    st.stop()

# ================================================================= load teams
teams = {}
if games:
    abbrevs = sorted({g["homeTeam"]["abbrev"] for g in games} | {g["awayTeam"]["abbrev"] for g in games})
    bar = st.progress(0.0, text="Loading rosters and stats…")
    for i, a in enumerate(abbrevs):
        try:
            teams[a] = team_model(a, cur_season, prev_season, date_str)
        except Exception:
            teams[a] = None
        bar.progress((i + 1) / len(abbrevs), text=f"Loading rosters and stats… {a}")
    bar.empty()
    odds = moneylines(date_str)
else:
    odds = {}

# ================================================================= sidebar (part 2)
starter_pick = {}
with st.sidebar:
    if teams:
        all_players = sorted({f"{s['name']} ({t['abbrev']})" for t in teams.values() if t
                              for s in t["skaters"]})
        excluded = set(st.multiselect("Exclude players (injuries, scratches)", all_players))
        with st.expander("Starting goalies"):
            st.caption("Default = most starts; backup if the team played yesterday. "
                       "Change to the confirmed starter.")
            for a, t in teams.items():
                if not t or not t["goalies"]:
                    continue
                names = [g["name"] for g in t["goalies"]]
                default = 1 if (t["b2b"] and len(names) > 1) else 0
                starter_pick[a] = st.selectbox(a, names, index=default, key=f"g_{a}")
    else:
        excluded = set()
    st.caption("Data: free NHL web API. Model: regressed team/player rates, goalie-adjusted "
               "Poisson goals, negative-binomial shots/saves. Not betting advice.")

if not games:
    tab_bt = st.tabs(["📈 Backtest"])[0]
    st.info(f"No NHL games on {pick_date:%A, %B %d, %Y}. You can still run the backtest.")
else:
    tab_daily, tab_picks, tab_games, tab_ml, tab_sog, tab_sv, tab_g, tab_bt = st.tabs(
        ["📋 Daily Card", "⭐ Picks", "🏟️ Games", "🎯 Moneyline", "🏒 Shots on Goal", "🧤 Goalie Saves", "🚨 Goals",
         "📈 Backtest"])

# ================================================================= project slate
slate, ml_rows, sog_rows, save_rows, goal_rows, picks = [], [], [], [], [], []


def add_pick(market, pick, game, team, image, p_model, price, price_note, edge, why, penalties=0):
    g = M.grade(edge * 100, penalties)
    if g:
        picks.append({"Grade": g, "Market": market, "Pick": pick, "Game": game, "Team": team,
                      "Image": image, "Model %": round(p_model * 100, 1),
                      "Fair": M.fmt_odds(M.to_american(p_model)), "Price": price,
                      "Price note": price_note, "Edge pts": round(edge * 100, 1), "Why": why})


def pick_starter(t):
    if not t or not t["goalies"]:
        return None, "none"
    chosen = starter_pick.get(t["abbrev"])
    default = t["goalies"][1] if (t["b2b"] and len(t["goalies"]) > 1) else t["goalies"][0]
    g = next((x for x in t["goalies"] if x["name"] == chosen), default)
    if g["name"] != default["name"]:
        return g, "Your pick"
    return g, ("B2B backup" if t["b2b"] and len(t["goalies"]) > 1 else "Most starts")


for g in games:
    home, away = g["homeTeam"], g["awayTeam"]
    h, a = teams.get(home["abbrev"]), teams.get(away["abbrev"])
    start = dt.datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00")).astimezone(tz)
    bx = boxscore(g["id"]) or {}
    label = f"{away['abbrev']} @ {home['abbrev']}"
    bh, ba = odds.get(g["id"], (None, None))
    info = {
        "label": label, "game": g, "start": start,
        "city": txt(bx.get("venueLocation")) or txt(home.get("placeName")),
        "venue": txt(g.get("venue")),
        "home_name": f"{txt(home.get('placeName'))} {txt(home.get('commonName'))}".strip(),
        "away_name": f"{txt(away.get('placeName'))} {txt(away.get('commonName'))}".strip(),
        "home_logo": home.get("darkLogo") or logo_url(home["abbrev"]),
        "away_logo": away.get("darkLogo") or logo_url(away["abbrev"]),
        "book": (bh, ba), "proj": None,
    }
    if h and a:
        hg, h_src = pick_starter(h)
        ag, a_src = pick_starter(a)
        h_sv = hg["sv"] if hg else M.LEAGUE["sv"]
        a_sv = ag["sv"] if ag else M.LEAGUE["sv"]
        p = M.project_game(h, a, h_sv, a_sv)
        info["proj"] = p
        info["starters"] = (ag["name"] if ag else "?", hg["name"] if hg else "?")

        nv = M.no_vig(bh, ba) if (bh is not None and ba is not None) else (None, None)
        for team, t, prob, bo, nvp, opp, side, lam, gl in (
            (home, h, p["ph"], bh, nv[0], away, "Home", p["lam_h"], hg),
            (away, a, 1 - p["ph"], ba, nv[1], home, "Away", p["lam_a"], ag),
        ):
            ml_rows.append({
                "Logo": info["home_logo"] if side == "Home" else info["away_logo"],
                "Team": team["abbrev"], "Opp": opp["abbrev"], "Side": side, "Game": label,
                "Win %": round(prob * 100, 1), "Fair ML": M.fmt_odds(M.to_american(prob)),
                "Book ML": M.fmt_odds(bo),
                "No-vig %": None if nvp is None else round(nvp * 100, 1),
                "Edge %": None if nvp is None else round((prob - nvp) * 100, 1),
                "Proj Goals": round(lam, 2), "Starter": gl["name"] if gl else "?",
                "Rest": "B2B" if t["b2b"] else "Rested",
                "Worth it at": M.worth_it_label(M.worth_it_price(prob)),
                "Grade": M.grade((prob - M.implied(bo)) * 100) if bo is not None else None,
            })
            if bo is not None:
                why = (f"Model {prob:.0%} vs market {nvp:.0%} (no-vig) · proj {lam:.1f} goals · "
                       f"{'opp starter ' + (ag if side == 'Home' else hg)['name'] if (ag if side == 'Home' else hg) else ''}"
                       f"{' · opp on B2B' if (a if side == 'Home' else h)['b2b'] else ''}")
                add_pick("Moneyline", f"{team['abbrev']} ML", label, team["abbrev"],
                         info["home_logo"] if side == "Home" else info["away_logo"],
                         prob, M.fmt_odds(bo), "DraftKings", prob - M.implied(bo), why,
                         penalties=1 if t["b2b"] and gl and len(t["goalies"]) > 1 else 0)

        for me, opp, opp_g, my_g, my_src, x_sog_against, venue in (
            (h, a, ag, hg, h_src, p["sog_a"], M.HOME_ADV),
            (a, h, hg, ag, a_src, p["sog_h"], 2 - M.HOME_ADV),
        ):
            opp_sv = opp_g["sv"] if opp_g else M.LEAGUE["sv"]
            shot_adj = opp["sa"] / M.LEAGUE["sog"]
            goal_adj = shot_adj * M.goalie_factor(opp_sv) * venue * (1 - M.B2B_PENALTY if me["b2b"] else 1)
            for s in me["skaters"]:
                if s["gp"] + s["prev_gp"] < min_gp:
                    continue
                if f"{s['name']} ({me['abbrev']})" in excluded:
                    continue
                if lineup_only and not s["in_lineup"]:
                    continue
                base = {"Photo": s["headshot"], "Player": s["name"], "Pos": s["pos"],
                        "Team": me["abbrev"], "Opp": opp["abbrev"], "Game": label,
                        "GP": s["gp"], "Prev GP": s["prev_gp"]}
                x_sog = s["sog_rate"] * shot_adj
                sog_rows.append({**base, "SOG/GP*": round(s["sog_rate"], 2), "Proj SOG": round(x_sog, 2),
                                 "O 1.5 %": round(M.p_at_least(2, x_sog, M.SOG_DISPERSION) * 100, 1),
                                 "O 2.5 %": round(M.p_at_least(3, x_sog, M.SOG_DISPERSION) * 100, 1),
                                 "O 3.5 %": round(M.p_at_least(4, x_sog, M.SOG_DISPERSION) * 100, 1)})
                x_g = s["g_rate"] * goal_adj
                p_any = M.p_at_least(1, x_g)
                goal_rows.append({**base, "G/GP*": round(s["g_rate"], 3), "Proj G": round(x_g, 3),
                                  "Anytime %": round(p_any * 100, 1),
                                  "Fair Odds": M.fmt_odds(M.to_american(p_any)),
                                  "2+ Goals %": round(M.p_at_least(2, x_g) * 100, 1)})

                small = 1 if s["gp"] + s["prev_gp"] < 30 else 0
                side, line, p, pb = M.prop_pick(s["sog_rate"], x_sog, M.SOG_DISPERSION)
                add_pick("Shots on Goal", f"{s['name']} {side} {line} SOG", label, me["abbrev"],
                         s["headshot"], p, M.fmt_odds(M.to_american(pb)), "est. line + price",
                         p - pb,
                         f"Proj {x_sog:.2f} vs usual {s['sog_rate']:.2f} · {opp['abbrev']} allows "
                         f"{opp['sa']:.1f} SOG/gm", small)
                sog_rows[-1].update({"Est line": line, "Lean": side,
                                     "Edge pts": round((p - pb) * 100, 1),
                                     "Grade": M.grade((p - pb) * 100, small)})

                p_base = 1 - math.exp(-s["g_rate"])
                book_p = min(0.95, p_base * M.ANYTIME_VIG)
                add_pick("Goals", f"{s['name']} anytime goal", label, me["abbrev"], s["headshot"],
                         p_any, M.fmt_odds(M.to_american(book_p)), "est. price",
                         p_any - book_p,
                         f"Proj {x_g:.2f} goals vs usual {s['g_rate']:.2f} · faces "
                         f"{opp_g['name'] if opp_g else 'unknown'} ({opp_sv:.3f})", small)
                goal_rows[-1].update({"Edge pts": round((p_any - book_p) * 100, 1),
                                      "Grade": M.grade((p_any - book_p) * 100, small),
                                      "Why": f"faces {opp_g['name'] if opp_g else 'unknown'} "
                                             f"({opp_sv:.3f}) · {opp['abbrev']} allows {opp['sa']:.1f} SOG/gm"})
            if my_g:
                x_sv = x_sog_against * my_g["sv"]
                save_rows.append({
                    "Photo": my_g["headshot"], "Goalie": my_g["name"], "Team": me["abbrev"],
                    "Opp": opp["abbrev"], "Game": label, "Pick": my_src,
                    "Starts (cur/prev)": f"{my_g['starts_cur']}/{my_g['starts_prev']}",
                    "SV%*": round(my_g["sv"], 3), "Opp Proj SOG": round(x_sog_against, 1),
                    "Proj Saves": round(x_sv, 1),
                    **{f"O {ln}.5 %": round(M.p_at_least(ln + 1, x_sv, M.SAVES_DISPERSION) * 100, 1)
                       for ln in (22, 25, 28)},
                })
                baseline = me["sa"] * my_g["sv"]
                side, line, p, pb = M.prop_pick(baseline, x_sv, M.SAVES_DISPERSION)
                add_pick("Goalie Saves", f"{my_g['name']} {side} {line} saves", label, me["abbrev"],
                         my_g["headshot"], p, M.fmt_odds(M.to_american(pb)), "est. line + price",
                         p - pb,
                         f"Proj {x_sv:.1f} saves vs usual {baseline:.1f} · {opp['abbrev']} proj "
                         f"{x_sog_against:.1f} SOG · {my_src.lower()}",
                         1 if my_src == "B2B backup" else 0)
                save_rows[-1].update({
                    "Usual": round(baseline, 1), "Est line": line, "Lean": side,
                    "Model %": round(p * 100, 1), "Edge pts": round((p - pb) * 100, 1),
                    "Grade": M.grade((p - pb) * 100, 1 if my_src == "B2B backup" else 0),
                    "Why": f"{opp['abbrev']} proj {x_sog_against:.1f} SOG (avg {M.LEAGUE['sog']:.1f}) · "
                           f"proj {x_sv:.1f} saves vs usual {baseline:.1f}"})
    slate.append(info)

IMG = st.column_config.ImageColumn

if games:
    pick_game = st.selectbox("Filter by game", ["All games"] + [s["label"] for s in slate])

    def filt(rows):
        df = pd.DataFrame(rows)
        if df.empty or pick_game == "All games":
            return df
        return df[df["Game"] == pick_game]

    # ------------------------------------------------------------- Daily Card
    with tab_daily:
        any_odds = any(s["book"][0] is not None for s in slate)
        st.subheader(f"Daily Card · {pick_date:%A, %B %d}")
        st.caption(f"{len(slate)} game(s) · rebuilt automatically from live NHL stats every time the app "
                   f"loads (refreshes every 10–30 min) · "
                   f"{'DraftKings moneylines posted' if any_odds else 'moneylines not posted yet — showing fair lines'}"
                   " · starters and prop lines unconfirmed until game day")

        def gtag(g):
            return f"<span class='grade g-{g[0]}' style='font-size:1rem;padding:2px 8px;min-width:0'>{g}</span>" if g else "<span style='opacity:.5'>—</span>"

        def section(title, rows_html, note=""):
            st.markdown(f"#### {title}")
            st.markdown("<div class='game-card' style='text-align:left'>" + "".join(rows_html) + "</div>",
                        unsafe_allow_html=True)
            if note:
                st.caption(note)

        def row(img, grade_html, title, sub):
            return (f"<div style='display:flex;gap:12px;align-items:center;padding:6px 0;"
                    f"border-bottom:1px solid #2a2f3a'>{grade_html}"
                    f"<img src='{img or ''}' style='width:40px;height:40px;border-radius:50%;object-fit:cover;"
                    f"background:#0e1117' onerror=\"this.style.visibility='hidden'\">"
                    f"<div style='flex:1;min-width:0'><b>{title}</b><div class='meta'>{sub}</div></div></div>")

        lines = [f"NHL PLUS Daily Card — {pick_date:%a %b %d, %Y}", ""]

        # Saves: every projected starter, best edge first
        sv = sorted(save_rows, key=lambda r: (M.GRADE_ORDER.get(r.get("Grade"), 9), -r.get("Edge pts", -99)))
        if sv:
            section("🧤 Goalie saves",
                    [row(r["Photo"], gtag(r.get("Grade")),
                         f"{r['Goalie']} ({r['Team']}) {r['Lean']} {r['Est line']} saves",
                         f"{r['Why']} · model {r['Model %']:.0f}% · edge {r['Edge pts']:+.1f} pts · {r['Pick'].lower()}")
                     for r in sv[:8]],
                    "Lines are estimated; confirm the starter and your book's line before betting.")
            lines += ["GOALIE SAVES"] + [f"  [{r.get('Grade') or '-'}] {r['Goalie']} ({r['Team']}) {r['Lean']} "
                                         f"{r['Est line']} — proj {r['Proj Saves']}" for r in sv[:8]] + [""]

        # SOG: top projections
        sg = sorted(sog_rows, key=lambda r: -r["Proj SOG"])[:8]
        if sg:
            section("🏒 Shots on goal — top candidates",
                    [row(r["Photo"], gtag(r.get("Grade")), f"{r['Player']} ({r['Team']} vs {r['Opp']})",
                         f"proj {r['Proj SOG']:.2f} · O1.5 {r['O 1.5 %']:.0f}% · O2.5 {r['O 2.5 %']:.0f}% · "
                         f"O3.5 {r['O 3.5 %']:.0f}% · est line {r['Est line']} ({r['Lean']} lean)")
                     for r in sg],
                    "Grade shows only when the matchup adds real edge over the player's usual line.")
            lines += ["SHOTS ON GOAL"] + [f"  [{r.get('Grade') or '-'}] {r['Player']} ({r['Team']}) proj "
                                          f"{r['Proj SOG']:.2f} · O2.5 {r['O 2.5 %']:.0f}% · O3.5 {r['O 3.5 %']:.0f}%"
                                          for r in sg] + [""]

        # Goals: top anytime
        gl_ = sorted(goal_rows, key=lambda r: -r["Anytime %"])[:8]
        if gl_:
            section("🚨 Anytime goal — top candidates",
                    [row(r["Photo"], gtag(r.get("Grade")), f"{r['Player']} ({r['Team']} vs {r['Opp']})",
                         f"{r['Anytime %']:.0f}% to score · fair {r['Fair Odds']} — bet only longer than this · "
                         f"{r['Why']}")
                     for r in gl_])
            lines += ["ANYTIME GOAL"] + [f"  [{r.get('Grade') or '-'}] {r['Player']} ({r['Team']}) "
                                         f"{r['Anytime %']:.0f}% · fair {r['Fair Odds']}" for r in gl_] + [""]

        # Moneyline
        mlr = []
        lines.append("MONEYLINE")
        for s in slate:
            p = s["proj"]
            if not p:
                continue
            g = s["game"]
            ab, hb = g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]
            bh, ba = s["book"]
            fav, pf = (hb, p["ph"]) if p["ph"] >= 0.5 else (ab, 1 - p["ph"])
            wa, wh = M.worth_it_price(1 - p["ph"]), M.worth_it_price(p["ph"])
            gh = M.grade((p["ph"] - M.implied(bh)) * 100) if bh is not None else None
            ga = M.grade((1 - p["ph"] - M.implied(ba)) * 100) if ba is not None else None
            best = gh or ga
            book = f" · book {ab} {M.fmt_odds(ba)} / {hb} {M.fmt_odds(bh)}" if bh is not None else ""
            pick_txt = (f" · <b>value: {hb if gh else ab}</b>" if best else "")
            mlr.append(row(s["home_logo"], gtag(best), f"{s['label']} · {s['start']:%I:%M %p %Z}",
                           f"model {fav} {pf:.0%} · fair {ab} {M.fmt_odds(M.to_american(1 - p['ph']))} / "
                           f"{hb} {M.fmt_odds(M.to_american(p['ph']))} · worth it at {ab} {M.worth_it_label(wa)}, "
                           f"{hb} {M.worth_it_label(wh)}{book}{pick_txt}"))
            lines.append(f"  {s['label']}: {fav} {pf:.0%} · worth it at {ab} {M.fmt_odds(wa)} / {hb} {M.fmt_odds(wh)}"
                         + (f" · book {ab} {M.fmt_odds(ba)} / {hb} {M.fmt_odds(bh)}" if bh is not None else ""))
        if mlr:
            section("🎯 Moneyline", mlr, "'Worth it at' = the worst price that still gives a B-grade (2-pt) edge.")

        lines += ["", "Model output, not guarantees. Bet responsibly — 1-800-GAMBLER."]
        st.download_button("⬇️ Download today's card", "\n".join(lines),
                           file_name=f"nhl-plus-{date_str}.txt", mime="text/plain")

    # ------------------------------------------------------------- Picks
    with tab_picks:
        st.subheader("Today's picks — graded")
        c1, c2, c3 = st.columns([2, 1, 1])
        markets = c1.multiselect("Markets", ["Moneyline", "Shots on Goal", "Goalie Saves", "Goals"],
                                 default=["Moneyline", "Shots on Goal", "Goalie Saves", "Goals"])
        min_grade = c2.selectbox("Minimum grade", [g for g, _ in M.GRADE_SCALE], index=3)
        per_market = c3.slider("Max per market", 1, 15, 5)
        df = filt(picks)
        if df.empty:
            st.info("No picks clear the edge threshold for this slate. That's normal on some nights — "
                    "passing is a pick too.")
        else:
            df = df[df["Market"].isin(markets) & (df["Grade"].map(M.GRADE_ORDER) <= M.GRADE_ORDER[min_grade])]
            df = (df.assign(_g=df["Grade"].map(M.GRADE_ORDER))
                    .sort_values(["_g", "Edge pts"], ascending=[True, False])
                    .groupby("Market", sort=False).head(per_market).drop(columns="_g"))
            if df.empty:
                st.info("No picks at this grade. Try lowering the minimum grade.")
            cols = st.columns(2)
            for i, r in enumerate(df.itertuples(index=False)):
                cls = "g-" + r.Grade[0]
                cols[i % 2].markdown(
                    f"""<div class="pick-card">
                          <div class="grade {cls}">{r.Grade}</div>
                          <img src="{r.Image or ''}" onerror="this.style.visibility='hidden'">
                          <div class="pick-main">
                            <div class="pick-title">{r.Pick}</div>
                            <div class="meta">{r.Market} · {r.Game} · model {r[6]:.1f}% · fair {r.Fair} ·
                              price {r.Price} ({r[9]}) · edge +{r[10]:.1f} pts</div>
                            <div class="pick-why">{r.Why}</div>
                          </div>
                        </div>""",
                    unsafe_allow_html=True)
            with st.expander("All picks as a table"):
                st.dataframe(df, hide_index=True, width="stretch",
                             column_config={"Image": IMG(width="small")})
        st.caption(
            "**How grades work:** edge = model probability minus the break-even probability at the "
            "price. A+ ≥ 7 pts · A ≥ 5 · B+ ≥ 3.5 · B ≥ 2 · C ≥ 0.5. Picks drop a grade for small "
            "player samples or a guessed backup goalie. Moneylines use the posted DraftKings price. "
            "Props use an **estimated** line and price (set from the player's usual average plus "
            "typical juice), so the edge is what the matchup adds — check your book's real line "
            "and price before betting. Grades rank model confidence; they don't guarantee results. Bet responsibly.")

    # ------------------------------------------------------------- Games
    with tab_games:
        st.subheader(f"{pick_date:%A, %B %d, %Y} · {len(slate)} game(s)")
        cols = st.columns(2)
        shown = 0
        for s in slate:
            if pick_game != "All games" and s["label"] != pick_game:
                continue
            g, p = s["game"], s["proj"]
            state = g.get("gameState", "FUT")
            score = ""
            if state in ("LIVE", "CRIT") + FINAL_STATES:
                score = f"<div class='ml'>{g['awayTeam'].get('score', 0)} – {g['homeTeam'].get('score', 0)}</div>"
            line = ""
            if p:
                ab, hb = g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]
                bh, ba = s["book"]
                book = (f"<div class='meta'>Book: {ab} {M.fmt_odds(ba)} · {hb} {M.fmt_odds(bh)}</div>"
                        if bh is not None else "")
                ta, th = teams[ab], teams[hb]
                rest = "".join(f"<span class='pill'>{x['abbrev']} B2B</span>" for x in (ta, th) if x["b2b"])
                line = (f"<div class='ml'>Model: {ab} {M.fmt_odds(M.to_american(1 - p['ph']))} · "
                        f"{hb} {M.fmt_odds(M.to_american(p['ph']))}</div>{book}"
                        f"<div class='meta'>Proj score {p['lam_a']:.1f} – {p['lam_h']:.1f} · "
                        f"Proj SOG {p['sog_a']:.0f} – {p['sog_h']:.0f}<br>"
                        f"🥅 {s['starters'][0]} vs {s['starters'][1]}</div>{rest}")
            tv = ", ".join(b.get("network", "") for b in g.get("tvBroadcasts", [])[:3])
            cols[shown % 2].markdown(
                f"""
                <div class="game-card">
                  <table style="width:100%;border:none"><tr style="border:none">
                    <td style="width:25%;border:none"><img src="{s['away_logo']}"><br><b>{s['away_name']}</b></td>
                    <td style="border:none">
                      <div class="matchup">{s['label']}</div>
                      <div class="meta">📅 {s['start']:%a, %b %d, %Y}<br>
                      🕒 {s['start']:%I:%M %p %Z}<br>
                      📍 {s['venue']}{', ' + s['city'] if s['city'] else ''}<br>
                      {('📺 ' + tv) if tv else ''}</div>
                      <span class="pill">{state}</span>{score}
                      <div style="margin-top:6px">{line}</div>
                    </td>
                    <td style="width:25%;border:none"><img src="{s['home_logo']}"><br><b>{s['home_name']}</b></td>
                  </tr></table>
                </div>
                """,
                unsafe_allow_html=True,
            )
            shown += 1

    # ------------------------------------------------------------- Moneyline
    with tab_ml:
        st.subheader("Moneyline — model vs. no-vig market")
        df = filt(ml_rows)
        if df.empty:
            st.info("No moneyline projections available.")
        else:
            if df["Book ML"].eq("—").all():
                st.caption("No partner odds posted for this date; compare **Fair ML** to your book.")
            else:
                st.caption("Edge = model win % minus the book's no-vig win % (bookmaker margin removed).")
            st.dataframe(df.sort_values("Win %", ascending=False), hide_index=True, width="stretch",
                         column_config={"Logo": IMG(width="small"),
                                        "Win %": st.column_config.ProgressColumn(
                                            min_value=0, max_value=100, format="%.1f%%")})

    # ------------------------------------------------------------- SOG
    with tab_sog:
        st.subheader("Shots on Goal — player projections")
        st.caption("*Rate regressed toward last season / position average. "
                   "Over % uses a negative binomial, which allows for the real game-to-game spread.")
        df = filt(sog_rows)
        if df.empty:
            st.info("No players match the current filters.")
        else:
            st.dataframe(df.sort_values("Proj SOG", ascending=False).head(show_n), hide_index=True,
                         width="stretch", column_config={"Photo": IMG(width="small")})

    # ------------------------------------------------------------- Saves
    with tab_sv:
        st.subheader("Goalie Saves — projected starters")
        st.caption("Change starters in the sidebar once they're confirmed. *SV% regressed toward league average.")
        df = filt(save_rows)
        if df.empty:
            st.info("No goalie data available.")
        else:
            st.dataframe(df.sort_values("Proj Saves", ascending=False), hide_index=True,
                         width="stretch", column_config={"Photo": IMG(width="small")})

    # ------------------------------------------------------------- Goals
    with tab_g:
        st.subheader("Goals — anytime goalscorer")
        st.caption("Adjusted for opponent shots allowed, opposing starter, home ice and rest.")
        df = filt(goal_rows)
        if df.empty:
            st.info("No players match the current filters.")
        else:
            st.dataframe(df.sort_values("Anytime %", ascending=False).head(show_n), hide_index=True,
                         width="stretch",
                         column_config={"Photo": IMG(width="small"),
                                        "Anytime %": st.column_config.ProgressColumn(
                                            min_value=0, max_value=60, format="%.1f%%")})

# ----------------------------------------------------------------- Backtest
with tab_bt:
    st.subheader("Backtest — moneyline model, walk-forward")
    st.caption("Each game is predicted only from results before it (no look-ahead). "
               "Covers the core goals model and home ice; historical odds and shot data aren't in "
               "the free feed, so this measures accuracy and calibration, not betting profit.")
    bt_season = st.selectbox("Season", [prev_season, cur_season],
                             format_func=lambda s: f"{s[:4]}–{s[4:]}")
    if st.button("Run backtest"):
        with st.spinner("Loading every team's results…"):
            res = season_results(bt_season)
        if len(res) < 20:
            st.info("Not enough completed games in that season yet.")
        else:
            rows = M.backtest(res)
            m = M.score_predictions(rows)
            c = st.columns(4)
            c[0].metric("Games", m["games"])
            c[1].metric("Accuracy", f"{m['accuracy']:.1%}", help="Picked side won")
            c[2].metric("Brier", f"{m['brier']:.4f}", f"{m['brier'] - 0.25:+.4f} vs coin flip",
                        delta_color="inverse")
            c[3].metric("Log loss", f"{m['log_loss']:.4f}", f"{m['log_loss'] - m['log_loss_coinflip']:+.4f} vs coin flip",
                        delta_color="inverse")
            st.caption(f"Home teams won {m['home_win_rate']:.1%} of games. Lower Brier and log loss is better.")
            cal = pd.DataFrame(M.calibration(rows))
            pts = alt.Chart(cal).mark_circle(size=120).encode(
                x=alt.X("predicted:Q", title="Predicted home win %", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("actual:Q", title="Actual home win %", scale=alt.Scale(domain=[0, 1])),
                tooltip=["bin", alt.Tooltip("predicted:Q", format=".1%"),
                         alt.Tooltip("actual:Q", format=".1%"), "games"])
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                strokeDash=[4, 4], color="gray").encode(x="x", y="y")
            st.altair_chart(diag + pts, width="stretch")
            st.caption("Dots on the dashed line = well calibrated.")

    with st.expander("Model notes"):
        st.markdown(f"""
- **Team rates** blend this season with a prior (last season pulled {M.PRIOR_REGRESS:.0%} toward league average), worth {M.TEAM_K} games.
- **Goals** = expected SOG × shooting % × opposing starter's save factor × home ice ({M.HOME_ADV}) × rest (B2B ±{M.B2B_PENALTY:.0%}) × last-10 form (±{M.FORM_CAP:.0%} max).
- **Goalie SV%** regressed with {M.GOALIE_SV_K} shots at league average ({M.LEAGUE['sv']:.3f}).
- **Shots / saves** use negative binomial (r = {M.SOG_DISPERSION:g} skaters, {M.SAVES_DISPERSION:g} saves); goals use Poisson.
- **Players** come from the current roster; newcomers' prior season is pulled from their player profile.
- No free injury feed exists: use the last-lineup filter and the exclude box.
""")

st.divider()
st.caption("NHL PLUS · For entertainment and research. Bet responsibly — 1-800-GAMBLER.")
