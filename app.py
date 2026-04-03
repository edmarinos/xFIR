import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import requests
from datetime import date, datetime, timezone
from supabase import create_client

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="xFIR — Expected First Inning Runs",
    page_icon="⚾",
    layout="wide"
)

# ── Supabase ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = get_supabase()

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    classifier = joblib.load(os.path.join(base, 'xfir_classifier.pkl'))
    regressor  = joblib.load(os.path.join(base, 'xfir_regressor.pkl'))
    scaler     = joblib.load(os.path.join(base, 'scaler_v2.pkl'))
    with open(os.path.join(base, 'features_v2.json')) as f:
        features = json.load(f)
    return classifier, regressor, scaler, features

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    pitchers = pd.read_csv(os.path.join(base, 'pitcher_list_2025.csv'))
    teams    = pd.read_csv(os.path.join(base, 'team_offense_2025.csv'))
    teams    = teams[teams['team'] != '- - -'].copy()
    return pitchers, teams

classifier, regressor, scaler, FEATURES = load_models()
pitchers_df, teams_df = load_data()

# ── Static data ───────────────────────────────────────────────────────────────
PARK_FACTORS = {
    'COL': 115, 'ATH': 112, 'CIN': 108, 'BAL': 107, 'DET': 106,
    'LAD': 106, 'BOS': 105, 'TOR': 105, 'KCR': 105, 'MIL': 104,
    'PHI': 104, 'MIN': 103, 'NYY': 103, 'ATL': 102, 'HOU': 102,
    'STL': 99,  'PIT': 98,  'MIA': 98,  'CHC': 97,  'NYM': 97,
    'SDP': 97,  'LAA': 96,  'SF':  96,  'ARI': 100, 'CLE': 95,
    'WSN': 95,  'CHW': 94,  'TBR': 94,  'SEA': 93,  'TEX': 93,
}

TEAM_NAMES = {
    'ARI': 'Arizona Diamondbacks',   'ATL': 'Atlanta Braves',
    'BAL': 'Baltimore Orioles',      'BOS': 'Boston Red Sox',
    'CHC': 'Chicago Cubs',           'CHW': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds',        'CLE': 'Cleveland Guardians',
    'COL': 'Colorado Rockies',       'DET': 'Detroit Tigers',
    'HOU': 'Houston Astros',         'KCR': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels',     'LAD': 'Los Angeles Dodgers',
    'MIA': 'Miami Marlins',          'MIL': 'Milwaukee Brewers',
    'MIN': 'Minnesota Twins',        'NYM': 'New York Mets',
    'NYY': 'New York Yankees',       'ATH': 'Oakland Athletics',
    'PHI': 'Philadelphia Phillies',  'PIT': 'Pittsburgh Pirates',
    'SDP': 'San Diego Padres',       'SEA': 'Seattle Mariners',
    'SF':  'San Francisco Giants',   'STL': 'St. Louis Cardinals',
    'TBR': 'Tampa Bay Rays',         'TEX': 'Texas Rangers',
    'TOR': 'Toronto Blue Jays',      'WSN': 'Washington Nationals'
}

MLB_NAME_TO_ABBR = {v: k for k, v in TEAM_NAMES.items()}
MLB_NAME_TO_ABBR['Athletics'] = 'ATH'

LEAGUE_AVG = {
    'sp_ERA': 4.20, 'sp_FIP': 4.10, 'sp_xFIP': 4.15, 'sp_SIERA': 4.10,
    'sp_K%': 0.225, 'sp_BB%': 0.082, 'sp_K-BB%': 0.143, 'sp_WHIP': 1.28,
    'sp_HR/9': 1.20, 'sp_GB%': 0.430, 'sp_SwStr%': 0.112, 'sp_CSW%': 0.278,
    'sp_HardHit%': 0.380, 'sp_xERA': 4.20
}

# ── Helper functions ──────────────────────────────────────────────────────────
def get_team_offense(team_abbr):
    row = teams_df[teams_df['team'] == team_abbr]
    if len(row) == 0:
        return {'team_OPS': 0.720, 'team_wRC': 100.0,
                'team_BB_pct': 0.082, 'team_K_pct': 0.225}
    return row.iloc[0].to_dict()

def get_pitcher_stats(pitcher_name):
    row = pitchers_df[pitchers_df['pitcher_name'] == pitcher_name]
    if len(row) == 0:
        return LEAGUE_AVG
    r = row.iloc[0]
    return {k: r[k] for k in ['sp_ERA', 'sp_FIP', 'sp_xFIP', 'sp_SIERA',
                                'sp_K%', 'sp_BB%', 'sp_K-BB%', 'sp_WHIP',
                                'sp_HR/9', 'sp_GB%', 'sp_SwStr%', 'sp_CSW%',
                                'sp_HardHit%', 'sp_xERA']}

def predict(batting_team, pitching_team, pitcher_stats, is_home, month):
    offense = get_team_offense(batting_team)
    park = PARK_FACTORS.get(batting_team if is_home else pitching_team, 100)
    input_dict = {
        **pitcher_stats,
        'is_home': int(is_home),
        'month': month,
        'team_rolling_runs': 0.30,
        'park_factor': park,
        'team_OPS':    offense.get('team_OPS', 0.720),
        'team_wRC':    offense.get('team_wRC', 100.0),
        'team_BB_pct': offense.get('team_BB_pct', 0.082),
        'team_K_pct':  offense.get('team_K_pct', 0.225),
    }
    input_df     = pd.DataFrame([input_dict])[FEATURES]
    input_scaled = scaler.transform(input_df)
    score_prob   = classifier.predict_proba(input_df)[0][1]
    exp_runs     = max(0, regressor.predict(input_scaled)[0])
    return score_prob, exp_runs

def american_to_implied(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def american_to_decimal(odds):
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def calculate_ev(prob, odds):
    profit = odds / 100 if odds > 0 else 100 / abs(odds)
    return (prob * profit) - ((1 - prob) * 1)

def match_pitcher(name):
    if name == 'TBD':
        return 'League Average'
    matches = pitchers_df[
        pitchers_df['pitcher_name'].str.lower() == name.lower()
    ]
    return matches.iloc[0]['pitcher_name'] if len(matches) > 0 else 'League Average'

def pitcher_stats_table(stats):
    return f"""
| Stat | Value |
|------|-------|
| ERA | {stats['sp_ERA']:.2f} |
| FIP | {stats['sp_FIP']:.2f} |
| xFIP | {stats['sp_xFIP']:.2f} |
| K% | {stats['sp_K%']:.1%} |
| BB% | {stats['sp_BB%']:.1%} |
| WHIP | {stats['sp_WHIP']:.2f} |
| HardHit% | {stats['sp_HardHit%']:.1%} |
| SwStr% | {stats['sp_SwStr%']:.1%} |
"""

# ── MLB API ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_todays_games(selected_date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={selected_date}&hydrate=probablePitcher"
    try:
        data = requests.get(url, timeout=10).json()
    except Exception:
        return []
    games = []
    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            away = game['teams']['away']
            home = game['teams']['home']
            away_name = away['team']['name']
            home_name = home['team']['name']
            away_abbr = MLB_NAME_TO_ABBR.get(away_name, away_name[:3].upper())
            home_abbr = MLB_NAME_TO_ABBR.get(home_name, home_name[:3].upper())
            away_pitcher = away.get('probablePitcher', {}).get('fullName', 'TBD')
            home_pitcher = home.get('probablePitcher', {}).get('fullName', 'TBD')
            game_time_utc = game.get('gameDate', '')
            try:
                from zoneinfo import ZoneInfo
                gt = datetime.fromisoformat(game_time_utc.replace('Z', '+00:00'))
                est = ZoneInfo('America/New_York')
                game_time_str = gt.astimezone(est).strftime('%-I:%M %p ET')
            except Exception:
                game_time_str = 'TBD'
            games.append({
                'game_pk':      game['gamePk'],
                'away_team':    away_abbr,
                'away_name':    away_name,
                'home_team':    home_abbr,
                'home_name':    home_name,
                'away_pitcher': away_pitcher,
                'home_pitcher': home_pitcher,
                'game_time':    game_time_str,
            })
    return games

# ── Supabase functions ────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_game_linescore(game_pk):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
    try:
        data = requests.get(url, timeout=10).json()
        innings = data.get('innings', [])
        if not innings:
            return None
        first_inning = innings[0]
        away_runs = first_inning.get('away', {}).get('runs', None)
        home_runs = first_inning.get('home', {}).get('runs', None)
        current_inning = data.get('currentInning', 0)
        is_final = current_inning >= 9 and not data.get('isTopInning', True)
        return {
            'away_runs_1st': away_runs,
            'home_runs_1st': home_runs,
            'total_runs_1st': (away_runs or 0) + (home_runs or 0),
            'nrfi': ((away_runs or 0) + (home_runs or 0)) == 0,
            'is_final': is_final,
            'current_inning': current_inning
        }
    except Exception:
        return None

def save_predictions_to_db(games, predictions_by_game):
    today = date.today().isoformat()
    selected = str(games[0].get('game_date', today)) if games else today
    
    # Only save predictions for today's actual games
    if str(selected_date) != today:
        return
        
    rows = []
    for g in games:
        pk = str(g['game_pk'])
        if pk in predictions_by_game:
            pred = predictions_by_game[pk]
            rows.append({
                'game_date':       today,
                'game_pk':         pk,
                'away_team':       g['away_team'],
                'home_team':       g['home_team'],
                'away_pitcher':    g['away_pitcher'],
                'home_pitcher':    g['home_pitcher'],
                'game_time':       g['game_time'],
                'nrfi_prob':       float(round(pred['nrfi_prob'], 4)),
                'yrfi_prob':       float(round(pred['yrfi_prob'], 4)),
                'outcome_nrfi':    None,
                'outcome_fetched': False
            })
    if rows:
        try:
            supabase.table('predictions').upsert(
                rows, on_conflict='game_date,game_pk'
            ).execute()
        except Exception as e:
            st.warning(f"Could not save predictions: {e}")

def fetch_and_update_outcomes():
    try:
        response = supabase.table('predictions')\
            .select('*')\
            .eq('outcome_fetched', False)\
            .execute()

        now_utc = datetime.now(timezone.utc)

        for row in response.data:
            try:
                game_date = row['game_date']
                game_dt = datetime.strptime(
                    f"{game_date}", '%Y-%m-%d'
                ).replace(tzinfo=timezone.utc)

                hours_since_midnight = (now_utc - game_dt).total_seconds() / 3600
                if hours_since_midnight < 4:
                    continue

            except Exception:
                continue

            result = fetch_game_linescore(row['game_pk'])
            if result and result['is_final']:
                supabase.table('predictions')\
                    .update({
                        'outcome_nrfi':    result['nrfi'],
                        'away_runs_1st':   result['away_runs_1st'],
                        'home_runs_1st':   result['home_runs_1st'],
                        'outcome_fetched': True
                    })\
                    .eq('id', row['id'])\
                    .execute()

    except Exception as e:
        st.warning(f"Could not fetch outcomes: {e}")

def manual_override(game_pk, game_date, nrfi_result):
    try:
        supabase.table('predictions')\
            .update({
                'outcome_nrfi':        nrfi_result,
                'outcome_fetched':     True,
                'manually_overridden': True
            })\
            .eq('game_pk', str(game_pk))\
            .eq('game_date', game_date)\
            .execute()
        st.success("Outcome saved.")
    except Exception as e:
        st.error(f"Could not save outcome: {e}")

def load_results(selected_date=None):
    try:
        query = supabase.table('predictions')\
            .select('*')\
            .not_.is_('outcome_nrfi', 'null')\
            .order('game_date', desc=True)
        if selected_date:
            query = query.eq('game_date', selected_date.isoformat())
        result = query.execute()
        return pd.DataFrame(result.data)
    except Exception:
        return pd.DataFrame()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚾ xFIR — Expected First Inning Runs")
st.markdown("Today's MLB games with first inning scoring predictions and EV calculator.")

# Date selector
col_prev, col_date, col_next = st.columns([1, 3, 1])

if 'selected_date' not in st.session_state:
    st.session_state.selected_date = date.today()

with col_prev:
    if st.button("◀ Previous Day", use_container_width=True):
        st.session_state.selected_date -= __import__('datetime').timedelta(days=1)

with col_date:
    st.session_state.selected_date = st.date_input(
        "Date",
        value=st.session_state.selected_date,
        label_visibility="collapsed"
    )

with col_next:
    if st.button("Next Day ▶", use_container_width=True):
        st.session_state.selected_date += __import__('datetime').timedelta(days=1)

selected_date = st.session_state.selected_date
today_str = selected_date.strftime('%A, %B %d %Y')
st.markdown(f"**{today_str}**")
st.markdown("---")

games = get_todays_games(selected_date)
month = selected_date.month

# Collect and save predictions
predictions_by_game = {}
for g in games:
    away = g['away_team']
    home = g['home_team']
    ap = match_pitcher(g['away_pitcher'])
    hp = match_pitcher(g['home_pitcher'])
    a_stats = get_pitcher_stats(ap) if ap != 'League Average' else LEAGUE_AVG
    h_stats = get_pitcher_stats(hp) if hp != 'League Average' else LEAGUE_AVG
    away_p, _ = predict(away, home, h_stats, is_home=False, month=month)
    home_p, _ = predict(home, away, a_stats, is_home=True, month=month)
    neither = (1 - away_p) * (1 - home_p)
    predictions_by_game[str(g['game_pk'])] = {
        'nrfi_prob': neither,
        'yrfi_prob': 1 - neither
    }

save_predictions_to_db(games, predictions_by_game)
fetch_and_update_outcomes()

if not games:
    st.warning("No games found for today. The MLB schedule may not be loaded yet.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📅 Today's Games", "🎰 Parlay Builder", "📊 Model Results"])

# ── Tab 1: Today's Games ──────────────────────────────────────────────────────
with tab1:
    st.markdown(f"### {len(games)} Games Today")
    pitcher_options = ['League Average'] + sorted(pitchers_df['pitcher_name'].tolist())

    for i, game in enumerate(games):
        away      = game['away_team']
        home      = game['home_team']
        away_name = game['away_name']
        home_name = game['home_name']
        game_time = game['game_time']

        away_pitcher_matched = match_pitcher(game['away_pitcher'])
        home_pitcher_matched = match_pitcher(game['home_pitcher'])

        with st.expander(
            f"⚾ {away_name} @ {home_name}  —  {game_time}  |  "
            f"{game['away_pitcher']} vs {game['home_pitcher']}",
            expanded=(i == 0)
        ):
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                away_pitcher_sel = st.selectbox(
                    f"✈️ Away Starter ({away})",
                    options=pitcher_options,
                    index=pitcher_options.index(away_pitcher_matched)
                          if away_pitcher_matched in pitcher_options else 0,
                    key=f"away_p_{i}"
                )
            with col_p2:
                home_pitcher_sel = st.selectbox(
                    f"🏠 Home Starter ({home})",
                    options=pitcher_options,
                    index=pitcher_options.index(home_pitcher_matched)
                          if home_pitcher_matched in pitcher_options else 0,
                    key=f"home_p_{i}"
                )

            away_stats = get_pitcher_stats(away_pitcher_sel) \
                         if away_pitcher_sel != 'League Average' else LEAGUE_AVG
            home_stats = get_pitcher_stats(home_pitcher_sel) \
                         if home_pitcher_sel != 'League Average' else LEAGUE_AVG

            away_prob, away_runs = predict(away, home, home_stats, is_home=False, month=month)
            home_prob, home_runs = predict(home, away, away_stats, is_home=True,  month=month)
            neither_prob = (1 - away_prob) * (1 - home_prob)
            either_prob  = 1 - neither_prob

            c1, c2, c3 = st.columns(3)
            with c1:
                emoji = "🟢" if away_prob >= 0.30 else "🟡" if away_prob >= 0.25 else "🔴"
                st.metric(f"{emoji} {away} Score (Top 1st)",
                          f"{away_prob:.1%}", f"xRuns: {away_runs:.3f}")
            with c2:
                emoji = "🟢" if home_prob >= 0.30 else "🟡" if home_prob >= 0.25 else "🔴"
                st.metric(f"{emoji} {home} Score (Bot 1st)",
                          f"{home_prob:.1%}", f"xRuns: {home_runs:.3f}")
            with c3:
                emoji = "🟢" if neither_prob >= 0.53 else "🔴"
                st.metric(f"{emoji} Scoreless 1st", f"{neither_prob:.1%}",
                          f"Either scores: {either_prob:.1%}")

            with st.expander("📋 Pitcher Stats"):
                ps1, ps2 = st.columns(2)
                with ps1:
                    st.markdown(f"**{home_pitcher_sel}** (pitching to {away})")
                    st.markdown(pitcher_stats_table(home_stats))
                with ps2:
                    st.markdown(f"**{away_pitcher_sel}** (pitching to {home})")
                    st.markdown(pitcher_stats_table(away_stats))

            st.markdown("#### 💰 EV Calculator")
            ev1, ev2 = st.columns(2)

            with ev1:
                st.markdown("**Scoreless 1st Inning (NRFI)**")
                odds_scoreless = st.number_input(
                    "Sportsbook Odds", value=-110, step=5,
                    key=f"odds_scoreless_{i}"
                )
                implied_s = american_to_implied(odds_scoreless)
                ev_s      = calculate_ev(neither_prob, odds_scoreless)
                edge_s    = neither_prob - implied_s
                st.metric("Model Prob",   f"{neither_prob:.1%}")
                st.metric("Implied Prob", f"{implied_s:.1%}",
                          delta=f"Edge: {edge_s:+.1%}")
                st.metric("EV per $100",  f"${ev_s * 100:.2f}",
                          delta="Profitable" if ev_s > 0 else "Unprofitable",
                          delta_color="normal" if ev_s > 0 else "inverse")

            with ev2:
                st.markdown("**At Least One Team Scores (YRFI)**")
                odds_scoring = st.number_input(
                    "Sportsbook Odds", value=-110, step=5,
                    key=f"odds_scoring_{i}"
                )
                implied_e = american_to_implied(odds_scoring)
                ev_e      = calculate_ev(either_prob, odds_scoring)
                edge_e    = either_prob - implied_e
                st.metric("Model Prob",   f"{either_prob:.1%}")
                st.metric("Implied Prob", f"{implied_e:.1%}",
                          delta=f"Edge: {edge_e:+.1%}")
                st.metric("EV per $100",  f"${ev_e * 100:.2f}",
                          delta="Profitable" if ev_e > 0 else "Unprofitable",
                          delta_color="normal" if ev_e > 0 else "inverse")

            if ev_s > 0.02:
                st.success(f"✅ NRFI looks +EV at those odds (edge: {edge_s:+.1%})")
            elif ev_e > 0.02:
                st.success(f"✅ YRFI looks +EV at those odds (edge: {edge_e:+.1%})")
            else:
                st.info("No strong edge detected at these odds. Try line shopping.")

            st.caption("⚠️ For educational purposes only. Not financial or betting advice.")

# ── Tab 2: Parlay Builder ─────────────────────────────────────────────────────
with tab2:
    st.subheader("🎰 Parlay Builder")
    st.caption("Build a parlay using today's games. Model probabilities are used to calculate true EV.")

    if 'num_legs' not in st.session_state:
        st.session_state.num_legs = 2

    col_add, col_remove, _ = st.columns([1, 1, 4])
    with col_add:
        if st.button("➕ Add Leg"):
            st.session_state.num_legs += 1
    with col_remove:
        if st.button("➖ Remove Leg") and st.session_state.num_legs > 2:
            st.session_state.num_legs -= 1

    st.markdown(f"**{st.session_state.num_legs} Leg Parlay**")

    game_options = [
        f"{g['away_name']} @ {g['home_name']} ({g['game_time']})"
        for g in games
    ]

    BET_TYPES = {
        'NRFI (Neither team scores)':    'nrfi',
        'YRFI (At least one scores)':    'yrfi',
        'Away team scores 1st inning':   'away_scores',
        'Home team scores 1st inning':   'home_scores',
    }

    parlay_legs = []

    for leg_num in range(st.session_state.num_legs):
        st.markdown(f"**Leg {leg_num + 1}**")
        lc1, lc2, lc3 = st.columns([3, 2, 1])

        with lc1:
            selected_game = st.selectbox(
                "Game", options=game_options,
                key=f"parlay_game_{leg_num}",
                label_visibility="collapsed"
            )
        with lc2:
            bet_type_label = st.selectbox(
                "Bet Type", options=list(BET_TYPES.keys()),
                key=f"parlay_bet_{leg_num}",
                label_visibility="collapsed"
            )
        with lc3:
            leg_odds = st.number_input(
                "Odds", value=-110, step=5,
                key=f"parlay_odds_{leg_num}",
                label_visibility="collapsed"
            )

        game_idx = game_options.index(selected_game)
        g    = games[game_idx]
        away = g['away_team']
        home = g['home_team']

        ap = match_pitcher(g['away_pitcher'])
        hp = match_pitcher(g['home_pitcher'])
        away_stats_p = get_pitcher_stats(ap) if ap != 'League Average' else LEAGUE_AVG
        home_stats_p = get_pitcher_stats(hp) if hp != 'League Average' else LEAGUE_AVG

        away_prob_p, _ = predict(away, home, home_stats_p, is_home=False, month=month)
        home_prob_p, _ = predict(home, away, away_stats_p, is_home=True,  month=month)
        neither_p = (1 - away_prob_p) * (1 - home_prob_p)
        either_p  = 1 - neither_p

        bet_key = BET_TYPES[bet_type_label]
        if bet_key == 'nrfi':
            leg_prob = neither_p
        elif bet_key == 'yrfi':
            leg_prob = either_p
        elif bet_key == 'away_scores':
            leg_prob = away_prob_p
        else:
            leg_prob = home_prob_p

        parlay_legs.append({
            'game': selected_game,
            'bet':  bet_type_label,
            'odds': leg_odds,
            'prob': leg_prob
        })

        st.caption(f"Model probability for this leg: **{leg_prob:.1%}**")
        st.markdown("")

    if len(parlay_legs) >= 2:
        st.markdown("---")
        st.markdown("### 📊 Parlay Summary")

        true_parlay_prob = 1.0
        for leg in parlay_legs:
            true_parlay_prob *= leg['prob']

        decimal_odds  = [american_to_decimal(leg['odds']) for leg in parlay_legs]
        parlay_decimal = 1.0
        for d in decimal_odds:
            parlay_decimal *= d

        parlay_payout   = parlay_decimal - 1
        parlay_american = int(parlay_payout * 100) if parlay_payout >= 1 \
                          else int(-100 / parlay_payout)
        parlay_ev       = (true_parlay_prob * parlay_payout) - (1 - true_parlay_prob)
        implied_parlay  = 1 / parlay_decimal

        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            st.metric("True Parlay Probability", f"{true_parlay_prob:.2%}")
        with pc2:
            st.metric("Implied Probability", f"{implied_parlay:.2%}")
        with pc3:
            st.metric("Combined Odds",
                      f"+{parlay_american}" if parlay_american > 0 else str(parlay_american))
        with pc4:
            st.metric("EV per $100", f"${parlay_ev * 100:.2f}",
                      delta="Profitable" if parlay_ev > 0 else "Unprofitable",
                      delta_color="normal" if parlay_ev > 0 else "inverse")

        leg_df = pd.DataFrame([{
            'Game':         leg['game'].split('(')[0].strip(),
            'Bet':          leg['bet'],
            'Odds':         leg['odds'],
            'Model Prob':   f"{leg['prob']:.1%}",
            'Implied Prob': f"{american_to_implied(leg['odds']):.1%}",
            'Leg Edge':     f"{(leg['prob'] - american_to_implied(leg['odds'])):+.1%}"
        } for leg in parlay_legs])
        st.dataframe(leg_df, use_container_width=True, hide_index=True)

        if parlay_ev > 0.05:
            st.success(f"✅ Strong +EV parlay. True prob ({true_parlay_prob:.2%}) > implied ({implied_parlay:.2%}).")
        elif parlay_ev > 0:
            st.info(f"📊 Slight +EV parlay ({parlay_ev * 100:.1f}%). Marginal edge.")
        else:
            st.warning(f"⚠️ Negative EV parlay ({parlay_ev * 100:.1f}%). Vig is eating your edge.")

        st.caption("⚠️ For educational purposes only. Not financial or betting advice.")

# ── Tab 3: Model Results ──────────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Model Performance Dashboard")
    st.caption("Tracking NRFI/YRFI predictions vs actual outcomes.")

    results_df = load_results(selected_date)

    if results_df.empty:
        st.info("No completed game results yet. Check back after today's games finish.")
    else:
        results_df['game_date'] = pd.to_datetime(results_df['game_date'])
        results_df['correct'] = (
            (results_df['outcome_nrfi'] == True)  & (results_df['nrfi_prob'] >= 0.5)
        ) | (
            (results_df['outcome_nrfi'] == False) & (results_df['nrfi_prob'] < 0.5)
        )

        total    = len(results_df)
        correct  = results_df['correct'].sum()
        nrfi_ct  = results_df['outcome_nrfi'].sum()
        avg_prob = results_df['nrfi_prob'].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Games Tracked",          total)
        m2.metric("Model Accuracy",         f"{correct/total:.1%}")
        m3.metric("Actual NRFI Rate",       f"{nrfi_ct/total:.1%}")
        m4.metric("Avg Predicted NRFI Prob",f"{avg_prob:.1%}")

        st.markdown("#### Calibration")
        results_df['prob_bucket'] = pd.cut(
            results_df['nrfi_prob'],
            bins=[0, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0],
            labels=['<45%', '45-50%', '50-55%', '55-60%', '60-65%', '>65%']
        )
        cal = results_df.groupby('prob_bucket', observed=True).agg(
            predicted=('nrfi_prob',   'mean'),
            actual=   ('outcome_nrfi','mean'),
            count=    ('outcome_nrfi','count')
        ).reset_index()
        st.dataframe(cal, use_container_width=True, hide_index=True)

        st.markdown("#### Game Log")
        display_df = results_df[[
            'game_date', 'away_team', 'home_team',
            'away_pitcher', 'home_pitcher',
            'nrfi_prob', 'yrfi_prob',
            'away_runs_1st', 'home_runs_1st',
            'outcome_nrfi', 'correct', 'manually_overridden'
        ]].copy()
        display_df['game_date'] = display_df['game_date'].dt.strftime('%Y-%m-%d')
        display_df['nrfi_prob'] = display_df['nrfi_prob'].apply(lambda x: f"{x:.1%}")
        display_df['yrfi_prob'] = display_df['yrfi_prob'].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("#### ✏️ Manual Override")
        st.caption("Use this if auto-fetch missed a game result.")
        try:
            pending_df = pd.DataFrame(
                supabase.table('predictions')
                .select('*')
                .eq('outcome_fetched', False)
                .eq('game_date', selected_date.isoformat())
                .execute().data
            )
        except Exception:
            pending_df = pd.DataFrame()

        if pending_df.empty:
            st.info("No pending games to override.")
        else:
            for _, row in pending_df.iterrows():
                oc1, oc2, oc3 = st.columns([3, 1, 1])
                with oc1:
                    st.write(f"{row['game_date']} — {row['away_team']} @ {row['home_team']}")
                with oc2:
                    if st.button("NRFI ✅", key=f"nrfi_yes_{row['id']}"):
                        manual_override(row['game_pk'], row['game_date'], True)
                        st.rerun()
                with oc3:
                    if st.button("YRFI ❌", key=f"nrfi_no_{row['id']}"):
                        manual_override(row['game_pk'], row['game_date'], False)
                        st.rerun()

# Footer stays outside tabs
st.markdown("---")
st.markdown("**Model:** XGBoost (classification) | Linear Regression (regression) | "
            "**Data:** Statcast 2022–2024 | **Schedule:** MLB Stats API")
st.markdown("**Classification AUC:** 0.521 | **Regression RMSE:** 1.045")
