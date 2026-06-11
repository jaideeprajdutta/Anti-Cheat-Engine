import pandas as pd

from confidence import (
    simulate_multiple_sessions,
    train_model,
    run_multisession_detection,
    BAN_THRESHOLD,
    SESSIONS_PER_PLAYER
)


def assign_lobbies(flag_counts, player_info):
    """
    Assign players to either the normal lobby or shadow lobby.

    Players are moved to the shadow lobby only if they are flagged
    repeatedly across multiple sessions.
    """
    lobby_rows = []

    for player_id, info in player_info.items():
        times_flagged = flag_counts.get(player_id, 0)

        if times_flagged >= BAN_THRESHOLD:
            lobby = "shadow_lobby"
            action = "silently_redirect"
        else:
            lobby = "normal_lobby"
            action = "allow"

        lobby_rows.append({
            "player_id": player_id,
            "player_name": info["player_name"],
            "is_cheater": info["is_cheater"],
            "cheat_type": info["cheat_type"],
            "times_flagged": times_flagged,
            "sessions_observed": SESSIONS_PER_PLAYER,
            "assigned_lobby": lobby,
            "action": action
        })

    return pd.DataFrame(lobby_rows)


def print_lobby_report(lobby_df):
    shadow_lobby = lobby_df[lobby_df["assigned_lobby"] == "shadow_lobby"]
    normal_lobby = lobby_df[lobby_df["assigned_lobby"] == "normal_lobby"]

    actual_cheaters = set(lobby_df[lobby_df["is_cheater"] == True]["player_id"])
    shadow_players = set(shadow_lobby["player_id"])

    true_positives = shadow_players & actual_cheaters
    false_positives = shadow_players - actual_cheaters
    false_negatives = actual_cheaters - shadow_players

    print("\n========================================")
    print("        SHADOW LOBBY ROUTING REPORT     ")
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
            "assigned_lobby"
        ]
        print(
            shadow_lobby[columns]
            .sort_values("times_flagged", ascending=False)
            .to_string(index=False)
        )

    print("\n--- ROUTING ACCURACY ---")
    print(f"Correctly shadowed cheaters (TP) : {len(true_positives)}")
    print(f"Innocent players shadowed (FP)   : {len(false_positives)}")
    print(f"Cheaters kept in normal lobby (FN): {len(false_negatives)}")

    precision = len(true_positives) / len(shadow_players) * 100 if shadow_players else 0
    recall = len(true_positives) / len(actual_cheaters) * 100 if actual_cheaters else 0

    print(f"\nPrecision : {precision:.1f}%")
    print(f"Recall    : {recall:.1f}%")


if __name__ == "__main__":
    print("Generating multi-session player behavior...")
    df = simulate_multiple_sessions()

    print("Training anomaly detection model...")
    model, scaler = train_model(df)

    print("Running confidence-based detection...")
    flag_counts, player_info = run_multisession_detection(df, model, scaler)

    print("Assigning players to lobbies...")
    lobby_df = assign_lobbies(flag_counts, player_info)

    print_lobby_report(lobby_df)

    lobby_df.to_csv("lobby_assignments.csv", index=False)
    print("\nLobby assignments saved to lobby_assignments.csv")
