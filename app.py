import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import requests
from datetime import date, datetime, timezone
from supabase import create_client
import anthropic

# ── Optional OpenAI fallback ──────────────────────────────────────────────────
try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

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
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={game_pk}"
        status_data = requests.get(url, timeout=10).json()
        dates = status_data.get('dates', [])
        if not dates:
            return None
        game_status    = dates[0]['games'][0].get('status', {})
        abstract_state = game_status.get('abstractGameState', '')
        if abstract_state != 'Final':
            return None
        url2 = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
        data = requests.get(url2, timeout=10).json()
        innings = data.get('innings', [])
        if not innings:
            return None
        first_inning = innings[0]
        away_runs = first_inning.get('away', {}).get('runs', None)
        home_runs = first_inning.get('home', {}).get('runs', None)
        if away_runs is None or home_runs is None:
            return None
        return {
            'away_runs_1st': away_runs,
            'home_runs_1st': home_runs,
            'total_runs_1st': away_runs + home_runs,
            'nrfi': (away_runs + home_runs) == 0,
            'is_final': True
        }
    except Exception:
        return None

def save_predictions_to_db(games, predictions_by_game, selected_date):
    today = date.today().isoformat()
    if str(selected_date) != today:
        return
    for g in games:
        pk = str(g['game_pk'])
        if pk not in predictions_by_game:
            continue
        pred = predictions_by_game[pk]
        try:
            existing = supabase.table('predictions')\
                .select('id')\
                .eq('game_date', today)\
                .eq('game_pk', pk)\
                .execute()
            if len(existing.data) > 0:
                continue
            supabase.table('predictions').insert({
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
            }).execute()
        except Exception:
            pass

def fetch_and_update_outcomes():
    try:
        response = supabase.table('predictions')\
            .select('*')\
            .eq('outcome_fetched', False)\
            .execute()

        now_utc = datetime.now(timezone.utc)

        # Remove postponed games
        for row in response.data:
            try:
                url = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={row['game_pk']}"
                status_data = requests.get(url, timeout=10).json()
                dates = status_data.get('dates', [])
                if not dates:
                    continue
                detailed_state = dates[0]['games'][0].get('status', {}).get('detailedState', '')
                if 'Postponed' in detailed_state:
                    supabase.table('predictions')\
                        .delete()\
                        .eq('game_pk', row['game_pk'])\
                        .eq('game_date', row['game_date'])\
                        .execute()
            except Exception:
                continue

        # Fetch outcomes for finished games
        for row in response.data:
            try:
                game_date = row['game_date']
                game_dt = datetime.strptime(game_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
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

def load_daily_odds(game_date):
    try:
        result = supabase.table('daily_odds')\
            .select('*')\
            .eq('game_date', game_date.isoformat())\
            .execute()
        return {row['game_pk']: row for row in result.data}
    except Exception:
        return {}

def save_daily_odds(game_date, game_pk, nrfi_odds, yrfi_odds):
    try:
        supabase.table('daily_odds').upsert({
            'game_date': game_date.isoformat(),
            'game_pk':   game_pk,
            'nrfi_odds': int(nrfi_odds),
            'yrfi_odds': int(yrfi_odds),
        }, on_conflict='game_date,game_pk').execute()
    except Exception:
        pass

def resolve_bankroll_bets():
    try:
        unresolved = supabase.table('bankroll')\
            .select('*')\
            .eq('resolved', False)\
            .execute()

        for bet in unresolved.data:
            if bet['is_parlay']:
                continue
            outcome = supabase.table('predictions')\
                .select('outcome_nrfi')\
                .eq('game_pk', bet['game_pk'])\
                .eq('game_date', bet['game_date'])\
                .execute()
            if not outcome.data or outcome.data[0]['outcome_nrfi'] is None:
                continue
            outcome_nrfi = outcome.data[0]['outcome_nrfi']
            bet_won      = (bet['bet_type'] == 'NRFI' and outcome_nrfi) or \
                           (bet['bet_type'] == 'YRFI' and not outcome_nrfi)
            profit_loss  = (bet['potential_payout'] - bet['bet_amount']) \
                           if bet_won else -bet['bet_amount']
            supabase.table('bankroll').update({
                'outcome_nrfi': outcome_nrfi,
                'bet_won':      bet_won,
                'profit_loss':  float(round(profit_loss, 2)),
                'resolved':     True
            }).eq('id', bet['id']).execute()

        parlay_bets = supabase.table('bankroll')\
            .select('*')\
            .eq('resolved', False)\
            .eq('is_parlay', True)\
            .execute()

        for parlay in parlay_bets.data:
            game_pks  = parlay['game_pk'].split('_')
            bet_types = parlay['bet_type'].split('+')
            game_date = parlay['game_date']
            all_resolved = True
            parlay_won   = True

            for pk, bt in zip(game_pks, bet_types):
                outcome = supabase.table('predictions')\
                    .select('outcome_nrfi')\
                    .eq('game_pk', pk)\
                    .eq('game_date', game_date)\
                    .execute()
                if not outcome.data or outcome.data[0]['outcome_nrfi'] is None:
                    all_resolved = False
                    break
                outcome_nrfi = outcome.data[0]['outcome_nrfi']
                leg_won = (bt == 'NRFI' and outcome_nrfi) or \
                          (bt == 'YRFI' and not outcome_nrfi)
                if not leg_won:
                    parlay_won = False

            if all_resolved:
                profit_loss = (parlay['potential_payout'] - parlay['bet_amount']) \
                              if parlay_won else -parlay['bet_amount']
                supabase.table('bankroll').update({
                    'bet_won':     parlay_won,
                    'profit_loss': float(round(profit_loss, 2)),
                    'resolved':    True
                }).eq('id', parlay['id']).execute()

        # Update bankroll history for ALL dates
        all_dates_result = supabase.table('bankroll')\
            .select('game_date')\
            .eq('resolved', True)\
            .execute()

        unique_dates = sorted(set(row['game_date'] for row in all_dates_result.data))

        for bet_date in unique_dates:
            resolved_bets = supabase.table('bankroll')\
                .select('*')\
                .eq('game_date', bet_date)\
                .eq('resolved', True)\
                .execute()
            if not resolved_bets.data:
                continue

            existing = supabase.table('bankroll_history')\
                .select('id, ending_bankroll, bets_placed')\
                .eq('game_date', bet_date)\
                .execute()

            # Count total bets for this date
            total_bets = supabase.table('bankroll')\
                .select('id')\
                .eq('game_date', bet_date)\
                .execute()

            resolved_count = len(resolved_bets.data)
            total_count    = len(total_bets.data)

            if existing.data and existing.data[0].get('bets_placed', 0) == total_count:
                continue

            daily_pl = sum(
                b['profit_loss'] for b in resolved_bets.data
                if b['profit_loss'] is not None
            )
            bets_won = sum(1 for b in resolved_bets.data if b['bet_won'])

            prev = supabase.table('bankroll_history')\
                .select('ending_bankroll')\
                .lt('game_date', bet_date)\
                .order('game_date', desc=True)\
                .limit(1)\
                .execute()

            start_br     = prev.data[0]['ending_bankroll'] if prev.data else 100.0
            new_bankroll = start_br + daily_pl

            supabase.table('bankroll_history').upsert({
                'game_date':         bet_date,
                'starting_bankroll': float(start_br),
                'ending_bankroll':   float(round(new_bankroll, 2)),
                'daily_pl':          float(round(daily_pl, 2)),
                'bets_placed':       len(resolved_bets.data),
                'bets_won':          bets_won
            }, on_conflict='game_date').execute()

    except Exception as e:
        st.warning(f"Could not resolve bets: {e}")

# ── AI Analyst ────────────────────────────────────────────────────────────────
_ANALYST_STAT_KEYS = [
    'sp_ERA', 'sp_FIP', 'sp_xFIP', 'sp_K%', 'sp_BB%',
    'sp_WHIP', 'sp_HardHit%', 'sp_SwStr%'
]

def _stats_to_tuple(stats):
    return tuple(stats.get(k, 0.0) for k in _ANALYST_STAT_KEYS)

@st.cache_data(ttl=3600, show_spinner=False)
def get_analyst_take(away_name, home_name, home_pitcher, away_pitcher,
                     home_stats_t, away_stats_t,
                     away_ops, away_wrc, home_ops, home_wrc,
                     away_prob, home_prob, away_runs, home_runs, park):
    """Call Claude Haiku to explain the prediction. Cached 1hr per game+pitchers."""
    home_stats = dict(zip(_ANALYST_STAT_KEYS, home_stats_t))
    away_stats = dict(zip(_ANALYST_STAT_KEYS, away_stats_t))
    scoreless = (1 - away_prob) * (1 - home_prob)

    prompt = f"""You are a sharp baseball analyst explaining a first-inning scoring prediction to a sports bettor.

MATCHUP: {away_name} @ {home_name}

{home_pitcher} pitching to {away_name} (top 1st):
ERA {home_stats['sp_ERA']:.2f} | FIP {home_stats['sp_FIP']:.2f} | K% {home_stats['sp_K%']:.1%} | BB% {home_stats['sp_BB%']:.1%} | WHIP {home_stats['sp_WHIP']:.2f} | HardHit% {home_stats['sp_HardHit%']:.1%}

{away_pitcher} pitching to {home_name} (bot 1st):
ERA {away_stats['sp_ERA']:.2f} | FIP {away_stats['sp_FIP']:.2f} | K% {away_stats['sp_K%']:.1%} | BB% {away_stats['sp_BB%']:.1%} | WHIP {away_stats['sp_WHIP']:.2f} | HardHit% {away_stats['sp_HardHit%']:.1%}

{away_name} offense: OPS {away_ops:.3f} | wRC+ {away_wrc:.0f}
{home_name} offense: OPS {home_ops:.3f} | wRC+ {home_wrc:.0f}
Park factor: {park} (100 = neutral, >100 = hitter-friendly)

MODEL OUTPUT:
- {away_name} scores top 1st: {away_prob:.1%} ({away_runs:.2f} xRuns)
- {home_name} scores bot 1st: {home_prob:.1%} ({home_runs:.2f} xRuns)
- Scoreless first inning: {scoreless:.1%}

In 3-4 sentences, explain WHY the model predicts what it predicts. Name the 2-3 stats that most drive the result. Call out any notable mismatch between pitcher quality and the opposing lineup. Be direct — no filler."""

    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    openai_key = st.secrets.get("OPENAI_API_KEY", "")

    if anthropic_key:
        try:
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text
        except anthropic.BadRequestError as e:
            if "credit balance" not in str(e):
                return f"⚠️ Anthropic error: {e}"
            # credit error — fall through to OpenAI if available

    if openai_key and _OPENAI_AVAILABLE:
        client = _openai.OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=256,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    return "⚠️ Credits depleted. Top up at console.anthropic.com/settings/billing or add OPENAI_API_KEY to secrets.toml."

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚾ xFIR — Expected First Inning Runs")
st.markdown("Today's MLB games with first inning scoring predictions and EV calculator.")

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

session_key = f"saved_{selected_date}"
if session_key not in st.session_state:
    save_predictions_to_db(games, predictions_by_game, selected_date)
    st.session_state[session_key] = True

fetch_and_update_outcomes()

if not games:
    st.warning("No games found for today. The MLB schedule may not be loaded yet.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📅 Today's Games", "🎰 Parlay Builder",
    "📊 Model Results", "💰 Financial"
])

# ── Tab 1: Today's Games ──────────────────────────────────────────────────────
with tab1:
    st.markdown(f"### {len(games)} Games Today")
    pitcher_options = ['League Average'] + sorted(pitchers_df['pitcher_name'].tolist())

    saved_odds_tab1 = load_daily_odds(selected_date)

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
                pk_str          = str(game['game_pk'])
                saved_t1        = saved_odds_tab1.get(pk_str, {})
                default_nrfi_t1 = saved_t1.get('nrfi_odds', -110)
                default_yrfi_t1 = saved_t1.get('yrfi_odds', -110)
            
                odds_scoreless = st.number_input(
                    "Sportsbook Odds", value=default_nrfi_t1, step=5,
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
                    "Sportsbook Odds", value=default_yrfi_t1, step=5,
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

            # AI Analyst
            st.markdown("#### 🤖 AI Analyst")
            if st.button("Generate Analysis", key=f"analyst_{i}"):
                away_offense = get_team_offense(away)
                home_offense = get_team_offense(home)
                park = PARK_FACTORS.get(home, 100)
                with st.spinner("Analyzing matchup..."):
                    take = get_analyst_take(
                        away_name, home_name,
                        home_pitcher_sel, away_pitcher_sel,
                        _stats_to_tuple(home_stats), _stats_to_tuple(away_stats),
                        away_offense.get('team_OPS', 0.720), away_offense.get('team_wRC', 100),
                        home_offense.get('team_OPS', 0.720), home_offense.get('team_wRC', 100),
                        away_prob, home_prob, away_runs, home_runs, park
                    )
                st.info(f"**AI Analyst:** {take}")

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
        'NRFI (Neither team scores)':  'nrfi',
        'YRFI (At least one scores)':  'yrfi',
        'Away team scores 1st inning': 'away_scores',
        'Home team scores 1st inning': 'home_scores',
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

        decimal_odds   = [american_to_decimal(leg['odds']) for leg in parlay_legs]
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

        all_results = load_results(selected_date=None)
        if not all_results.empty:
            all_results['correct'] = (
                (all_results['outcome_nrfi'] == True)  & (all_results['nrfi_prob'] >= 0.5)
            ) | (
                (all_results['outcome_nrfi'] == False) & (all_results['nrfi_prob'] < 0.5)
            )
            total_all   = len(all_results)
            correct_all = all_results['correct'].sum()
        else:
            total_all   = 0
            correct_all = 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Games Tracked",        total)
        m2.metric("Today's Accuracy",     f"{correct/total:.1%}")
        m3.metric("Total Model Accuracy", f"{correct_all/total_all:.1%}" if total_all > 0 else "N/A",
                  delta=f"{total_all} games total")
        m4.metric("Actual NRFI Rate",     f"{nrfi_ct/total:.1%}")
        m5.metric("Avg Predicted NRFI",   f"{avg_prob:.1%}")

        st.markdown("#### Calibration")
        results_df['prob_bucket'] = pd.cut(
            results_df['nrfi_prob'],
            bins=[0, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0],
            labels=['<45%', '45-50%', '50-55%', '55-60%', '60-65%', '>65%']
        )
        cal = results_df.groupby('prob_bucket', observed=True).agg(
            predicted=('nrfi_prob',    'mean'),
            actual=   ('outcome_nrfi', 'mean'),
            count=    ('outcome_nrfi', 'count')
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
            today_iso  = date.today().isoformat()
            pending_df = pd.DataFrame(
                supabase.table('predictions')
                .select('*')
                .eq('outcome_fetched', False)
                .lt('game_date', today_iso)
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

# ── Tab 4: Financial ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("💰 Bankroll Manager")
    st.caption("Enter today's NRFI/YRFI odds from your sportsbook. The model will recommend the top 5 bets and a 2-leg parlay using fractional Kelly sizing.")

    resolve_bankroll_bets()

    try:
        bankroll_result = supabase.table('bankroll_history')\
            .select('ending_bankroll, game_date')\
            .order('game_date', desc=True)\
            .limit(1)\
            .execute()
        if bankroll_result.data:
            current_bankroll = bankroll_result.data[0]['ending_bankroll']
            last_date        = bankroll_result.data[0]['game_date']
            st.success(f"Current bankroll: **${current_bankroll:.2f}** (as of {last_date})")
        else:
            current_bankroll = None
    except Exception:
        current_bankroll = None

    if current_bankroll is None:
        st.info("No bankroll history found. Set your starting bankroll below.")
        starting_bankroll = st.number_input(
            "Starting Bankroll ($)", value=100.0, step=10.0, format="%.2f"
        )
        if st.button("Set Starting Bankroll"):
            try:
                supabase.table('bankroll_history').insert({
                    'game_date':         date.today().isoformat(),
                    'starting_bankroll': starting_bankroll,
                    'ending_bankroll':   starting_bankroll,
                    'daily_pl':          0.0,
                    'bets_placed':       0,
                    'bets_won':          0
                }).execute()
                st.success(f"Bankroll set to ${starting_bankroll:.2f}")
                st.rerun()
            except Exception as e:
                st.error(f"Could not set bankroll: {e}")
        st.stop()

    st.markdown("---")

    # Load saved odds
    saved_odds = load_daily_odds(selected_date)

    st.subheader("📋 Enter Today's Odds")
    st.caption("Odds are saved automatically and persist through the day.")

    odds_inputs = {}

    for i, game in enumerate(games):
        away      = game['away_team']
        home      = game['home_team']
        game_time = game['game_time']
        pk        = str(game['game_pk'])

        saved        = saved_odds.get(pk, {})
        default_nrfi = saved.get('nrfi_odds', -115)
        default_yrfi = saved.get('yrfi_odds', -105)

        col_label, col_nrfi, col_yrfi = st.columns([3, 1, 1])
        with col_label:
            st.markdown(f"**{away} @ {home}** — {game_time}")
        with col_nrfi:
            nrfi_odds = st.number_input(
                "NRFI", value=default_nrfi, step=5,
                key=f"fin_nrfi_{i}",
                label_visibility="visible"
            )
        with col_yrfi:
            yrfi_odds = st.number_input(
                "YRFI", value=default_yrfi, step=5,
                key=f"fin_yrfi_{i}",
                label_visibility="visible"
            )

        odds_inputs[pk] = {'nrfi_odds': nrfi_odds, 'yrfi_odds': yrfi_odds}

        if nrfi_odds != default_nrfi or yrfi_odds != default_yrfi:
            save_daily_odds(selected_date, pk, nrfi_odds, yrfi_odds)

    st.caption("✅ Odds auto-save when changed.")
    st.markdown("---")

    KELLY_FRACTION = 0.25
    MIN_EDGE       = 0.03

    def kelly_bet(prob, odds, bankroll, fraction=KELLY_FRACTION):
        b     = american_to_decimal(odds) - 1
        q     = 1 - prob
        kelly = max(0, (b * prob - q) / b)
        bet   = kelly * fraction * bankroll
        return round(bet, 2), round(kelly * 100, 2)

    # ── Build EV-based bets ───────────────────────────────────────────────────
    all_bets = []
    for i, game in enumerate(games):
        pk   = str(game['game_pk'])
        away = game['away_team']
        home = game['home_team']

        if pk not in predictions_by_game:
            continue

        nrfi_prob = float(predictions_by_game[pk]['nrfi_prob'])
        yrfi_prob = float(predictions_by_game[pk]['yrfi_prob'])
        nrfi_odds = odds_inputs[pk]['nrfi_odds']
        yrfi_odds = odds_inputs[pk]['yrfi_odds']

        nrfi_ev   = calculate_ev(nrfi_prob, nrfi_odds)
        yrfi_ev   = calculate_ev(yrfi_prob, yrfi_odds)
        nrfi_edge = nrfi_prob - american_to_implied(nrfi_odds)
        yrfi_edge = yrfi_prob - american_to_implied(yrfi_odds)

        if nrfi_ev >= yrfi_ev:
            best_side, best_prob, best_odds, best_ev, best_edge = \
                'NRFI', nrfi_prob, nrfi_odds, nrfi_ev, nrfi_edge
        else:
            best_side, best_prob, best_odds, best_ev, best_edge = \
                'YRFI', yrfi_prob, yrfi_odds, yrfi_ev, yrfi_edge

        if best_edge >= MIN_EDGE:
            bet_amount, kelly_pct = kelly_bet(best_prob, best_odds, current_bankroll)
            all_bets.append({
                'game_pk':          pk,
                'away_team':        away,
                'home_team':        home,
                'game_time':        game['game_time'],
                'bet_type':         best_side,
                'model_prob':       best_prob,
                'odds':             best_odds,
                'implied_prob':     american_to_implied(best_odds),
                'edge':             best_edge,
                'ev':               best_ev,
                'kelly_pct':        kelly_pct,
                'bet_amount':       bet_amount,
                'potential_payout': round(bet_amount * american_to_decimal(best_odds), 2),
            })

    all_bets = sorted(all_bets, key=lambda x: x['ev'], reverse=True)
    top5 = all_bets[:5]
    top2 = all_bets[:2]

    # ── Build probability-based bets ──────────────────────────────────────────
    prob_bets = []
    for i, game in enumerate(games):
        pk   = str(game['game_pk'])
        away = game['away_team']
        home = game['home_team']

        if pk not in predictions_by_game:
            continue

        nrfi_prob = float(predictions_by_game[pk]['nrfi_prob'])
        yrfi_prob = float(predictions_by_game[pk]['yrfi_prob'])
        nrfi_odds = odds_inputs[pk]['nrfi_odds']
        yrfi_odds = odds_inputs[pk]['yrfi_odds']

        if nrfi_prob >= yrfi_prob:
            best_side, best_prob, best_odds = 'NRFI', nrfi_prob, nrfi_odds
        else:
            best_side, best_prob, best_odds = 'YRFI', yrfi_prob, yrfi_odds

        best_ev   = calculate_ev(best_prob, best_odds)
        best_edge = best_prob - american_to_implied(best_odds)
        bet_amount, kelly_pct = kelly_bet(best_prob, best_odds, current_bankroll)

        prob_bets.append({
            'game_pk':          pk,
            'away_team':        away,
            'home_team':        home,
            'game_time':        game['game_time'],
            'bet_type':         best_side,
            'model_prob':       best_prob,
            'odds':             best_odds,
            'implied_prob':     american_to_implied(best_odds),
            'edge':             best_edge,
            'ev':               best_ev,
            'kelly_pct':        kelly_pct,
            'bet_amount':       bet_amount,
            'potential_payout': round(bet_amount * american_to_decimal(best_odds), 2),
        })

    prob_bets  = sorted(prob_bets, key=lambda x: x['model_prob'], reverse=True)
    top5_prob  = prob_bets[:5]
    top2_prob  = prob_bets[:2]

    def bet_table(bets):
        return pd.DataFrame([{
            'Game':       f"{b['away_team']} @ {b['home_team']}",
            'Time':       b['game_time'],
            'Bet':        b['bet_type'],
            'Odds':       b['odds'],
            'Model Prob': f"{b['model_prob']:.1%}",
            'Implied':    f"{b['implied_prob']:.1%}",
            'Edge':       f"{b['edge']:+.1%}",
            'EV':         f"${b['ev']*100:.2f}",
            'Kelly %':    f"{b['kelly_pct']:.1f}%",
            'Bet $':      f"${b['bet_amount']:.2f}",
            'To Win':     f"${b['potential_payout'] - b['bet_amount']:.2f}"
        } for b in bets])

    def parlay_summary(top2_list, label):
        if len(top2_list) < 2:
            return
        pp       = top2_list[0]['model_prob'] * top2_list[1]['model_prob']
        pd_      = american_to_decimal(top2_list[0]['odds']) * american_to_decimal(top2_list[1]['odds'])
        payout   = pd_ - 1
        p_am     = int(payout * 100) if payout >= 1 else int(-100 / payout)
        p_ev     = (pp * payout) - (1 - pp)
        p_kelly, _ = kelly_bet(pp, p_am, current_bankroll)
        st.markdown(f"**🎰 {label}**")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Parlay Odds",      f"+{p_am}" if p_am > 0 else str(p_am))
        pc2.metric("True Probability", f"{pp:.2%}")
        pc3.metric("EV per $100",      f"${p_ev*100:.2f}")
        pc4.metric("Kelly Bet Size",   f"${p_kelly:.2f}")
        for leg in top2_list:
            st.markdown(f"- **{leg['away_team']} @ {leg['home_team']}** — {leg['bet_type']} ({leg['odds']})")

    # ── Display EV bets ───────────────────────────────────────────────────────
    if not top5:
        st.warning("No bets meet the minimum 3% edge threshold today. No EV recommendations.")
    else:
        st.subheader(f"🎯 Top {len(top5)} Recommended Bets — EV Strategy")
        st.caption("Highest edge over the implied market probability.")
        st.dataframe(bet_table(top5), use_container_width=True, hide_index=True)
        if len(top2) >= 2:
            st.markdown("---")
            parlay_summary(top2, "EV Parlay (Top 2 EV Bets)")

    # ── Display Probability bets ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"🔵 Top {len(top5_prob)} Recommended Bets — Probability Strategy")
    st.caption("Highest model confidence in one outcome, regardless of odds.")
    st.dataframe(bet_table(top5_prob), use_container_width=True, hide_index=True)
    if len(top2_prob) >= 2:
        st.markdown("---")
        parlay_summary(top2_prob, "Probability Parlay (Top 2 Confidence Bets)")

    # ── Place Bets ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("✅ Place Today's Bets")
    st.caption("Both EV and Probability bets are placed simultaneously.")

    today_bets_result = supabase.table('bankroll')\
        .select('id')\
        .eq('game_date', date.today().isoformat())\
        .eq('is_parlay', False)\
        .execute()

    if today_bets_result.data:
        st.info("✅ Bets already placed for today. Check back after games finish for results.")
    else:
        if st.button("✅ Confirm & Place Bets", type="primary"):
            try:
                today = date.today().isoformat()
                rows  = []

                # EV strategy bets
                for b in top5:
                    rows.append({
                        'game_date':        today,
                        'game_pk':          b['game_pk'],
                        'away_team':        b['away_team'],
                        'home_team':        b['home_team'],
                        'bet_type':         b['bet_type'],
                        'model_prob':       float(b['model_prob']),
                        'odds':             int(b['odds']),
                        'implied_prob':     float(b['implied_prob']),
                        'edge':             float(b['edge']),
                        'ev':               float(b['ev']),
                        'kelly_pct':        float(b['kelly_pct']),
                        'bet_amount':       float(b['bet_amount']),
                        'potential_payout': float(b['potential_payout']),
                        'is_parlay':        False,
                        'resolved':         False,
                        'strategy':         'ev',
                    })

                if len(top2) >= 2:
                    ap_prob    = top2[0]['model_prob'] * top2[1]['model_prob']
                    ap_decimal = american_to_decimal(top2[0]['odds']) * american_to_decimal(top2[1]['odds'])
                    ap_payout  = ap_decimal - 1
                    ap_am      = int(ap_payout * 100) if ap_payout >= 1 else int(-100 / ap_payout)
                    ap_ev      = (ap_prob * ap_payout) - (1 - ap_prob)
                    ap_kelly, ap_kelly_pct = kelly_bet(ap_prob, ap_am, current_bankroll)
                    rows.append({
                        'game_date':        today,
                        'game_pk':          f"{top2[0]['game_pk']}_{top2[1]['game_pk']}",
                        'away_team':        f"{top2[0]['away_team']}+{top2[1]['away_team']}",
                        'home_team':        f"{top2[0]['home_team']}+{top2[1]['home_team']}",
                        'bet_type':         f"{top2[0]['bet_type']}+{top2[1]['bet_type']}",
                        'model_prob':       float(ap_prob),
                        'odds':             int(ap_am),
                        'implied_prob':     float(1 / ap_decimal),
                        'edge':             float(ap_prob - (1 / ap_decimal)),
                        'ev':               float(ap_ev),
                        'kelly_pct':        float(ap_kelly_pct),
                        'bet_amount':       float(ap_kelly),
                        'potential_payout': float(ap_kelly * ap_decimal),
                        'is_parlay':        True,
                        'parlay_id':        f"parlay_{today}_ev",
                        'resolved':         False,
                        'strategy':         'ev',
                    })

                # Probability strategy bets
                for b in top5_prob:
                    rows.append({
                        'game_date':        today,
                        'game_pk':          b['game_pk'],
                        'away_team':        b['away_team'],
                        'home_team':        b['home_team'],
                        'bet_type':         b['bet_type'],
                        'model_prob':       float(b['model_prob']),
                        'odds':             int(b['odds']),
                        'implied_prob':     float(b['implied_prob']),
                        'edge':             float(b['edge']),
                        'ev':               float(b['ev']),
                        'kelly_pct':        float(b['kelly_pct']),
                        'bet_amount':       float(b['bet_amount']),
                        'potential_payout': float(b['potential_payout']),
                        'is_parlay':        False,
                        'resolved':         False,
                        'strategy':         'prob',
                    })

                if len(top2_prob) >= 2:
                    pp_prob    = top2_prob[0]['model_prob'] * top2_prob[1]['model_prob']
                    pp_decimal = american_to_decimal(top2_prob[0]['odds']) * american_to_decimal(top2_prob[1]['odds'])
                    pp_payout  = pp_decimal - 1
                    pp_am      = int(pp_payout * 100) if pp_payout >= 1 else int(-100 / pp_payout)
                    pp_ev      = (pp_prob * pp_payout) - (1 - pp_prob)
                    pp_kelly, pp_kelly_pct = kelly_bet(pp_prob, pp_am, current_bankroll)
                    rows.append({
                        'game_date':        today,
                        'game_pk':          f"{top2_prob[0]['game_pk']}_{top2_prob[1]['game_pk']}",
                        'away_team':        f"{top2_prob[0]['away_team']}+{top2_prob[1]['away_team']}",
                        'home_team':        f"{top2_prob[0]['home_team']}+{top2_prob[1]['home_team']}",
                        'bet_type':         f"{top2_prob[0]['bet_type']}+{top2_prob[1]['bet_type']}",
                        'model_prob':       float(pp_prob),
                        'odds':             int(pp_am),
                        'implied_prob':     float(1 / pp_decimal),
                        'edge':             float(pp_prob - (1 / pp_decimal)),
                        'ev':               float(pp_ev),
                        'kelly_pct':        float(pp_kelly_pct),
                        'bet_amount':       float(pp_kelly),
                        'potential_payout': float(pp_kelly * pp_decimal),
                        'is_parlay':        True,
                        'parlay_id':        f"parlay_{today}_prob",
                        'resolved':         False,
                        'strategy':         'prob',
                    })

                supabase.table('bankroll').insert(rows).execute()
                st.success(f"✅ {len(rows)} bets placed ({len(top5)+1} EV + {len(top5_prob)+1} Probability)!")
                st.rerun()

            except Exception as e:
                st.error(f"Could not place bets: {e}")

    # ── Bankroll History ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Bankroll History")

    try:
        history = supabase.table('bankroll_history')\
            .select('*')\
            .order('game_date', desc=False)\
            .execute()

        if history.data:
            hist_df = pd.DataFrame(history.data)

            bh1, bh2, bh3, bh4 = st.columns(4)
            bh1.metric("Starting Bankroll",
                       f"${hist_df.iloc[0]['starting_bankroll']:.2f}")
            bh2.metric("Current Bankroll",
                       f"${hist_df.iloc[-1]['ending_bankroll']:.2f}",
                       delta=f"${hist_df['daily_pl'].sum():.2f} total")
            bh3.metric("Total Bets",  hist_df['bets_placed'].sum())
            bh4.metric("Total Wins",  hist_df['bets_won'].sum())

            # Strategy comparison
            try:
                all_bets_df = pd.DataFrame(
                    supabase.table('bankroll')
                    .select('game_date, strategy, profit_loss, bet_won, resolved')
                    .eq('resolved', True)
                    .execute().data
                )

                if not all_bets_df.empty and 'strategy' in all_bets_df.columns:
                    all_bets_df['game_date'] = pd.to_datetime(all_bets_df['game_date']).dt.date

                    ev_bets        = all_bets_df[all_bets_df['strategy'] == 'ev'].copy()
                    prob_bets_hist = all_bets_df[all_bets_df['strategy'] == 'prob'].copy()

                    ev_daily   = ev_bets.groupby('game_date')['profit_loss'].sum()
                    prob_daily = prob_bets_hist.groupby('game_date')['profit_loss'].sum()

                    # Get all unique dates across both strategies
                    all_dates = sorted(set(ev_daily.index) | set(prob_daily.index))

                    chart_df = pd.DataFrame(index=all_dates)
                    chart_df['EV Strategy']          = ev_daily.reindex(all_dates).fillna(0).cumsum()
                    chart_df['Probability Strategy'] = prob_daily.reindex(all_dates).fillna(0).cumsum()
                    chart_df.index = chart_df.index.astype(str)

                    st.markdown("#### 📊 Strategy Comparison — Cumulative P&L")

                    # Summary metrics side by side
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.markdown("**🎯 EV Strategy**")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Total P&L",    f"${ev_bets['profit_loss'].sum():.2f}" if not ev_bets.empty else "$0.00")
                        col_b.metric("Win Rate",     f"{ev_bets['bet_won'].mean():.1%}" if not ev_bets.empty else "N/A")
                        col_c.metric("Bets Placed",  len(ev_bets) if not ev_bets.empty else 0)
                    with sc2:
                        st.markdown("**🔵 Probability Strategy**")
                        col_d, col_e, col_f = st.columns(3)
                        col_d.metric("Total P&L",    f"${prob_bets_hist['profit_loss'].sum():.2f}" if not prob_bets_hist.empty else "$0.00")
                        col_e.metric("Win Rate",     f"{prob_bets_hist['bet_won'].mean():.1%}" if not prob_bets_hist.empty else "N/A")
                        col_f.metric("Bets Placed",  len(prob_bets_hist) if not prob_bets_hist.empty else 0)

                    st.line_chart(chart_df, use_container_width=True)

                    # Daily breakdown table
                    st.markdown("#### Daily P&L by Strategy")
                    daily_breakdown = pd.DataFrame({
                        'Date':                 [str(d) for d in all_dates],
                        'EV P&L':               [f"${ev_daily.get(d, 0):.2f}" for d in all_dates],
                        'Prob P&L':             [f"${prob_daily.get(d, 0):.2f}" for d in all_dates],
                        'EV Cumulative':        [f"${chart_df.loc[str(d), 'EV Strategy']:.2f}" for d in all_dates],
                        'Prob Cumulative':      [f"${chart_df.loc[str(d), 'Probability Strategy']:.2f}" for d in all_dates],
                    })
                    st.dataframe(daily_breakdown, use_container_width=True, hide_index=True)

            except Exception:
                pass

            st.markdown("#### 📈 Overall Bankroll")
            st.line_chart(
                hist_df.set_index('game_date')['ending_bankroll'],
                use_container_width=True
            )

            st.markdown("#### Daily Log")
            display_hist = hist_df[[
                'game_date', 'starting_bankroll', 'ending_bankroll',
                'daily_pl', 'bets_placed', 'bets_won'
            ]].copy()
            display_hist.columns = ['Date', 'Start', 'End', 'P&L', 'Bets', 'Wins']
            display_hist['Start'] = display_hist['Start'].apply(lambda x: f"${x:.2f}")
            display_hist['End']   = display_hist['End'].apply(lambda x: f"${x:.2f}")
            display_hist['P&L']   = display_hist['P&L'].apply(
                lambda x: f"+${x:.2f}" if x > 0 else f"-${abs(x):.2f}"
            )
            st.dataframe(display_hist, use_container_width=True, hide_index=True)

        else:
            st.info("No bankroll history yet. Place your first bets above.")

    except Exception as e:
        st.warning(f"Could not load bankroll history: {e}")

    st.caption("⚠️ For educational purposes only. Not financial or betting advice.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Model:** XGBoost (classification) | Linear Regression (regression) | "
            "**Data:** Statcast 2022–2024 | **Schedule:** MLB Stats API")
st.markdown("**Classification AUC:** 0.521 | **Regression RMSE:** 1.045")
