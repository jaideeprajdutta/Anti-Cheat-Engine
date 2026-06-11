import random
import numpy as np
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# -----------------------------------------------
# This function generates one session of data
# for a single player based on their player type
# -----------------------------------------------

def generate_session(player_id, player_name, is_cheater, cheat_type=None):
    
    if not is_cheater:
        # Normal player behavior - all values are realistic
        avg_velocity      = round(random.uniform(3.0, 6.0), 2)
        max_velocity      = round(avg_velocity + random.uniform(0.5, 2.0), 2)
        accuracy          = round(random.uniform(0.20, 0.60), 2)
        actions_per_sec   = round(random.uniform(1.0, 3.5), 2)
        score_per_min     = round(random.uniform(10, 80), 2)
        headshot_rate     = round(random.uniform(0.05, 0.40), 2)
        session_duration  = round(random.uniform(15, 60), 2)

    else:
        # Cheater behavior - values are abnormal based on cheat type
        if cheat_type == "speedhack":
            avg_velocity      = round(random.uniform(15.0, 25.0), 2)
            max_velocity      = round(avg_velocity + random.uniform(5.0, 10.0), 2)
            accuracy          = round(random.uniform(0.20, 0.55), 2)
            actions_per_sec   = round(random.uniform(1.0, 3.5), 2)
            score_per_min     = round(random.uniform(80, 200), 2)
            headshot_rate     = round(random.uniform(0.05, 0.35), 2)
            session_duration  = round(random.uniform(10, 40), 2)

        elif cheat_type == "aimbot":
            avg_velocity      = round(random.uniform(3.0, 6.0), 2)
            max_velocity      = round(avg_velocity + random.uniform(0.5, 2.0), 2)
            accuracy          = round(random.uniform(0.92, 1.00), 2)
            actions_per_sec   = round(random.uniform(1.0, 3.5), 2)
            score_per_min     = round(random.uniform(150, 400), 2)
            headshot_rate     = round(random.uniform(0.85, 1.00), 2)
            session_duration  = round(random.uniform(20, 60), 2)

        elif cheat_type == "scorefarmer":
            avg_velocity      = round(random.uniform(3.0, 6.0), 2)
            max_velocity      = round(avg_velocity + random.uniform(0.5, 2.0), 2)
            accuracy          = round(random.uniform(0.20, 0.55), 2)
            actions_per_sec   = round(random.uniform(8.0, 15.0), 2)
            score_per_min     = round(random.uniform(500, 1000), 2)
            headshot_rate     = round(random.uniform(0.05, 0.35), 2)
            session_duration  = round(random.uniform(5, 20), 2)

    return {
        "player_id"       : player_id,
        "player_name"     : player_name,
        "is_cheater"      : is_cheater,
        "cheat_type"      : cheat_type if is_cheater else "none",
        "avg_velocity"    : avg_velocity,
        "max_velocity"    : max_velocity,
        "accuracy"        : accuracy,
        "actions_per_sec" : actions_per_sec,
        "score_per_min"   : score_per_min,
        "headshot_rate"   : headshot_rate,
        "session_duration": session_duration
    }


# -----------------------------------------------
# Generate 200 players: 190 normal, 10 cheaters
# -----------------------------------------------

def generate_all_players():
    players = []
    cheat_types = ["speedhack", "aimbot", "scorefarmer"]

    for i in range(1, 201):
        player_id   = f"P{str(i).zfill(4)}"
        player_name = fake.user_name()

        if i <= 190:
            session = generate_session(player_id, player_name, is_cheater=False)
        else:
            cheat_type = cheat_types[(i - 191) % 3]
            session = generate_session(player_id, player_name, is_cheater=True, cheat_type=cheat_type)

        players.append(session)

    return players


if __name__ == "__main__":
    import pandas as pd

    all_players = generate_all_players()
    df = pd.DataFrame(all_players)

    print("\n--- SAMPLE OF NORMAL PLAYERS ---")
    print(df[df.is_cheater == False].head(5).to_string(index=False))

    print("\n--- SAMPLE OF CHEATERS ---")
    print(df[df.is_cheater == True].to_string(index=False))

    df.to_csv("player_data.csv", index=False)
    print("\n--- player_data.csv saved successfully ---")