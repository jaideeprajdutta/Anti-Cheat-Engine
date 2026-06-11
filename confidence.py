import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------
# This file simulates running the detection
# engine across MULTIPLE game sessions per player
# and only banning players who are consistently
# flagged. One bad session does not get you banned.
# This is exactly how EA's real system works.
# -----------------------------------------------

FEATURES = [
    "avg_velocity",
    "max_velocity",
    "accuracy",
    "actions_per_sec",
    "score_per_min",
    "headshot_rate",
    "session_duration"
]

# How many sessions we simulate per player
SESSIONS_PER_PLAYER = 5

# How many flags needed across sessions to get banned
BAN_THRESHOLD = 3


# -----------------------------------------------
# Import the simulator to generate fresh sessions
# -----------------------------------------------

from simulator import generate_all_players, generate_session
import random
import faker as fk

fake = fk.Faker()


def simulate_multiple_sessions(n_sessions=SESSIONS_PER_PLAYER):
    """
    Generate multiple sessions for each player.
    Cheaters cheat consistently. Normal players
    may occasionally have unusual stats but will
    not be flagged repeatedly across sessions.
    """
    all_sessions = []

    base_players = generate_all_players()

    for session_num in range(n_sessions):
        for player in base_players:
            pid        = player["player_id"]
            pname      = player["player_name"]
            is_cheater = player["is_cheater"]
            cheat_type = player["cheat_type"] if is_cheater else None

            session = generate_session(
                pid, pname, is_cheater, cheat_type
            )
            session["session_num"] = session_num + 1
            all_sessions.append(session)

    return pd.DataFrame(all_sessions)


def train_model(df):
    normal = df[df["is_cheater"] == False]
    X      = normal[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_scaled)
    return model, scaler


def run_multisession_detection(df, model, scaler):
    """
    Run detection on every session.
    Track how many times each player gets flagged.
    Only ban players flagged in 3 or more sessions.
    """
    flag_counts    = defaultdict(int)
    player_info    = {}

    for session_num in df["session_num"].unique():
        session_df  = df[df["session_num"] == session_num].copy()
        X           = session_df[FEATURES]
        X_scaled    = scaler.transform(X)
        preds       = model.predict(X_scaled)
        scores      = model.decision_function(X_scaled)

        session_df["flagged"]         = preds == -1
        session_df["suspicion_score"] = np.round(-scores, 4)

        for _, row in session_df.iterrows():
            pid = row["player_id"]
            if row["flagged"]:
                flag_counts[pid] += 1
            if pid not in player_info:
                player_info[pid] = {
                    "player_name" : row["player_name"],
                    "is_cheater"  : row["is_cheater"],
                    "cheat_type"  : row["cheat_type"]
                }

    return flag_counts, player_info


def print_final_report(flag_counts, player_info):
    print("\n========================================")
    print("   MULTI-SESSION CONFIDENCE REPORT      ")
    print(f"   Sessions per player : {SESSIONS_PER_PLAYER}")
    print(f"   Ban threshold       : {BAN_THRESHOLD}+ flags")
    print("========================================\n")

    banned          = []
    cleared         = []
    actual_cheaters = []
    actual_innocent = []

    for pid, info in player_info.items():
        flags     = flag_counts.get(pid, 0)
        is_banned = flags >= BAN_THRESHOLD

        entry = {
            "player_id"   : pid,
            "player_name" : info["player_name"],
            "cheat_type"  : info["cheat_type"],
            "times_flagged": flags,
            "is_cheater"  : info["is_cheater"],
            "banned"      : is_banned
        }

        if is_banned:
            banned.append(entry)
        else:
            cleared.append(entry)

        if info["is_cheater"]:
            actual_cheaters.append(pid)
        else:
            actual_innocent.append(pid)

    banned_ids  = set(e["player_id"] for e in banned)
    cheater_ids = set(actual_cheaters)

    true_positives  = banned_ids & cheater_ids
    false_positives = banned_ids - cheater_ids
    false_negatives = cheater_ids - banned_ids

    print(f"Total players    : {len(player_info)}")
    print(f"Banned           : {len(banned)}")
    print(f"Cleared          : {len(cleared)}")

    print("\n--- BANNED PLAYERS ---")
    for e in sorted(banned, key=lambda x: -x["times_flagged"]):
        label = "CHEATER" if e["is_cheater"] else "FALSE POSITIVE"
        print(f"  {e['player_id']} | {e['player_name']:<20} | "
              f"Flagged {e['times_flagged']}/{SESSIONS_PER_PLAYER} sessions | "
              f"{e['cheat_type']:<12} | {label}")

    print("\n--- DETECTION ACCURACY ---")
    print(f"Actual cheaters          : {len(cheater_ids)}")
    print(f"Correctly banned    (TP) : {len(true_positives)}")
    print(f"Wrongly banned      (FP) : {len(false_positives)}")
    print(f"Cheaters missed     (FN) : {len(false_negatives)}")

    precision = (len(true_positives) / len(banned_ids) * 100) if banned_ids else 0
    recall    = (len(true_positives) / len(cheater_ids) * 100) if cheater_ids else 0

    print(f"\nPrecision : {precision:.1f}%  "
          f"(of banned players, how many were actual cheaters)")
    print(f"Recall    : {recall:.1f}%  "
          f"(of all cheaters, how many did we catch)")


if __name__ == "__main__":
    print("Simulating 5 sessions for 200 players...")
    df             = simulate_multiple_sessions()

    print("Training model on normal player behavior...")
    model, scaler  = train_model(df)

    print("Running multi-session detection...")
    flag_counts, player_info = run_multisession_detection(df, model, scaler)

    print_final_report(flag_counts, player_info)
    