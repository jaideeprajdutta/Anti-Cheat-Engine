from simulator import generate_session
from detector import train_model, detect_cheaters, print_report
import pandas as pd
import random
from faker import Faker

fake = Faker()


def generate_custom_players(total_players, cheater_count):
    players = []
    cheat_types = ["speedhack", "aimbot", "scorefarmer"]

    normal_count = total_players - cheater_count

    for i in range(1, total_players + 1):
        player_id = f"P{str(i).zfill(4)}"
        player_name = fake.user_name()

        if i <= normal_count:
            session = generate_session(
                player_id,
                player_name,
                is_cheater=False
            )
        else:
            cheat_type = random.choice(cheat_types)
            session = generate_session(
                player_id,
                player_name,
                is_cheater=True,
                cheat_type=cheat_type
            )

        players.append(session)

    return pd.DataFrame(players)


if __name__ == "__main__":
    print("ANTI-CHEAT AGENT")
    print("================")

    total_players = int(input("Enter total number of players: "))
    cheater_count = int(input("Enter number of cheaters: "))

    if cheater_count >= total_players:
        print("Cheater count must be less than total players.")
        exit()

    df = generate_custom_players(total_players, cheater_count)

    df.to_csv("player_data.csv", index=False)

    print("\nTraining model...")
    model, scaler = train_model(df)

    print("Running detection...")
    results = detect_cheaters(df, model, scaler)

    print_report(results)

    results.to_csv("detection_results.csv", index=False)

    print("\nResults saved to detection_results.csv")
