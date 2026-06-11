import random
from collections import defaultdict

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from simulator import generate_session


fake = Faker()

FEATURES = [
    "avg_velocity",
    "max_velocity",
    "accuracy",
    "actions_per_sec",
    "score_per_min",
    "headshot_rate",
    "session_duration"
]


def generate_base_players(total_players, cheater_count):
    players = []
    cheat_types = ["speedhack", "aimbot", "scorefarmer"]

    normal_count = total_players - cheater_count

    for i in range(1, total_players + 1):
        player_id = f"P{str(i).zfill(4)}"
        player_name = fake.user_name()

        if i <= normal_count:
            player = {
                "player_id": player_id,
                "player_name": player_name,
                "is_cheater": False,
                "cheat_type": "none"
            }
        else:
            player = {
                "player_id": player_id,
                "player_name": player_name,
                "is_cheater": True,
                "cheat_type": random.choice(cheat_types)
            }

        players.append(player)

    return players


def simulate_dynamic_sessions(total_players, cheater_count, sessions_per_player):
    all_sessions = []
    base_players = generate_base_players(total_players, cheater_count)

    for session_num in range(1, sessions_per_player + 1):
        for player in base_players:
            cheat_type = player["cheat_type"] if player["is_cheater"] else None

            session = generate_session(
                player["player_id"],
                player["player_name"],
                player["is_cheater"],
                cheat_type
            )

            session["session_num"] = session_num
            all_sessions.append(session)

    return pd.DataFrame(all_sessions)


def train_model(df):
    normal_players = df[df["is_cheater"] == False]
    X_train = normal_players[FEATURES]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    model.fit(X_train_scaled)

    print(f"Model trained successfully on {len(normal_players)} normal sessions.")

    return model, scaler


def run_detection_across_sessions(df, model, scaler):
    flag_counts = defaultdict(int)
    player_info = {}

    for session_num in sorted(df["session_num"].unique()):
        session_df = df[df["session_num"] == session_num].copy()

        X = session_df[FEATURES]
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)

        session_df["flagged"] = predictions == -1
        session_df["suspicion_score"] = np.round(-scores, 4)

        for _, row in session_df.iterrows():
            player_id = row["player_id"]

            if row["flagged"]:
                flag_counts[player_id] += 1

            if player_id not in player_info:
                player_info[player_id] = {
                    "player_name": row["player_name"],
                    "is_cheater": row["is_cheater"],
                    "cheat_type": row["cheat_type"]
                }

    return flag_counts, player_info


def assign_lobbies(flag_counts, player_info, ban_threshold, sessions_per_player):
    lobby_rows = []

    for player_id, info in player_info.items():
        times_flagged = flag_counts.get(player_id, 0)

        if times_flagged >= ban_threshold:
            assigned_lobby = "shadow_lobby"
            action = "silently_redirect"
        else:
            assigned_lobby = "normal_lobby"
            action = "allow"

        lobby_rows.append({
            "player_id": player_id,
            "player_name": info["player_name"],
            "is_cheater": info["is_cheater"],
            "cheat_type": info["cheat_type"],
            "times_flagged": times_flagged,
            "sessions_observed": sessions_per_player,
            "assigned_lobby": assigned_lobby,
            "action": action
        })

    return pd.DataFrame(lobby_rows)


def print_report(lobby_df):
    shadow_lobby = lobby_df[lobby_df["assigned_lobby"] == "shadow_lobby"]
    normal_lobby = lobby_df[lobby_df["assigned_lobby"] == "normal_lobby"]

    actual_cheaters = set(lobby_df[lobby_df["is_cheater"] == True]["player_id"])
    shadow_players = set(shadow_lobby["player_id"])

    true_positives = shadow_players & actual_cheaters
    false_positives = shadow_players - actual_cheaters
    false_negatives = actual_cheaters - shadow_players

    print("\n========================================")
    print("     DYNAMIC SHADOW LOBBY REPORT        ")
    print("========================================")
    print(f"Total players analyzed : {len(lobby_df)}")
    print(f"Normal lobby players   : {len(normal_lobby)}")
    print(f"Shadow lobby players   : {len(shadow_lobby)}")

    print("\n--- SHADOW LOBBY PLAYERS ---")

    if shadow_lobby.empty:
        print("No players were moved to the shadow lobby.")
    else:
        columns = [
            "player_id",
            "player_name",
            "cheat_type",
            "times_flagged",
            "sessions_observed",
            "assigned_lobby"
        ]

        print(
            shadow_lobby[columns]
            .sort_values("times_flagged", ascending=False)
            .to_string(index=False)
        )

    print("\n--- ROUTING ACCURACY ---")
    print(f"Actual cheaters                  : {len(actual_cheaters)}")
    print(f"Correctly shadowed cheaters (TP) : {len(true_positives)}")
    print(f"Innocent players shadowed (FP)   : {len(false_positives)}")
    print(f"Cheaters kept normal lobby (FN)  : {len(false_negatives)}")

    precision = len(true_positives) / len(shadow_players) * 100 if shadow_players else 0
    recall = len(true_positives) / len(actual_cheaters) * 100 if actual_cheaters else 0

    print(f"\nPrecision : {precision:.1f}%")
    print(f"Recall    : {recall:.1f}%")


if __name__ == "__main__":
    print("DYNAMIC ANTI-CHEAT SHADOW LOBBY")
    print("================================")

    total_players = int(input("Enter total number of players: "))
    cheater_count = int(input("Enter number of cheaters: "))
    sessions_per_player = int(input("Enter number of sessions per player: "))
    ban_threshold = int(input("Enter ban threshold: "))

    if total_players <= 0:
        print("Total players must be greater than 0.")
        exit()

    if cheater_count < 0 or cheater_count >= total_players:
        print("Cheater count must be at least 0 and less than total players.")
        exit()

    if sessions_per_player <= 0:
        print("Sessions per player must be greater than 0.")
        exit()

    if ban_threshold <= 0 or ban_threshold > sessions_per_player:
        print("Ban threshold must be between 1 and the number of sessions.")
        exit()

    print("\nGenerating dynamic multi-session data...")
    df = simulate_dynamic_sessions(
        total_players,
        cheater_count,
        sessions_per_player
    )

    df.to_csv("dynamic_player_sessions.csv", index=False)

    print("Training anomaly detection model...")
    model, scaler = train_model(df)

    print("Running detection across sessions...")
    flag_counts, player_info = run_detection_across_sessions(df, model, scaler)

    print("Assigning players to lobbies...")
    lobby_df = assign_lobbies(
        flag_counts,
        player_info,
        ban_threshold,
        sessions_per_player
    )

    print_report(lobby_df)

    lobby_df.to_csv("dynamic_lobby_assignments.csv", index=False)
    print("\nDynamic session data saved to dynamic_player_sessions.csv")
    print("Dynamic lobby assignments saved to dynamic_lobby_assignments.csv")
