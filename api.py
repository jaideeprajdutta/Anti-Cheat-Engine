from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from simulator import generate_all_players

app = FastAPI(
    title="Anti-Cheat Detection Engine",
    description="Behavioral anomaly detection system for multiplayer games",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = [
    "avg_velocity",
    "max_velocity",
    "accuracy",
    "actions_per_sec",
    "score_per_min",
    "headshot_rate",
    "session_duration"
]

model                 = None
scaler                = None
player_flag_counts    = defaultdict(int)
player_session_counts = defaultdict(int)
player_registry       = {}
BAN_THRESHOLD         = 3


def train_on_startup():
    global model, scaler
    print("Training model on normal player behavior...")
    df      = pd.DataFrame(generate_all_players())
    normal  = df[df["is_cheater"] == False]
    X       = normal[FEATURES]
    sc      = StandardScaler()
    X_sc    = sc.fit_transform(X)
    m       = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    m.fit(X_sc)
    model  = m
    scaler = sc
    print("Model ready. API is live.")


train_on_startup()


class PlayerSession(BaseModel):
    player_id        : str
    player_name      : str
    avg_velocity     : float
    max_velocity     : float
    accuracy         : float
    actions_per_sec  : float
    score_per_min    : float
    headshot_rate    : float
    session_duration : float


class DetectionResult(BaseModel):
    player_id        : str
    player_name      : str
    flagged          : bool
    suspicion_score  : float
    times_flagged    : int
    sessions_observed: int
    verdict          : str
    lobby            : str


@app.get("/")
def root():
    return {
        "message": "Anti-Cheat Detection Engine is running",
        "version": "1.0.0",
        "status" : "online"
    }


@app.post("/analyze")
def analyze_player(session: PlayerSession):
    global model, scaler
    global player_flag_counts, player_session_counts, player_registry

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    features = np.array([[
        session.avg_velocity,
        session.max_velocity,
        session.accuracy,
        session.actions_per_sec,
        session.score_per_min,
        session.headshot_rate,
        session.session_duration
    ]])

    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    raw_score       = model.decision_function(features_scaled)[0]
    suspicion_score = round(float(-raw_score), 4)
    flagged         = bool(prediction == -1)

    pid = session.player_id
    player_session_counts[pid] += 1
    player_registry[pid]        = session.player_name

    if flagged:
        player_flag_counts[pid] += 1

    times_flagged     = int(player_flag_counts[pid])
    sessions_observed = int(player_session_counts[pid])
    is_banned         = times_flagged >= BAN_THRESHOLD

    if is_banned:
        verdict = "BANNED"
        lobby   = "shadow_lobby"
    elif flagged:
        verdict = "SUSPICIOUS"
        lobby   = "normal_lobby"
    else:
        verdict = "CLEAN"
        lobby   = "normal_lobby"

    return {
        "player_id"        : pid,
        "player_name"      : session.player_name,
        "flagged"          : flagged,
        "suspicion_score"  : suspicion_score,
        "times_flagged"    : times_flagged,
        "sessions_observed": sessions_observed,
        "verdict"          : verdict,
        "lobby"            : lobby
    }


@app.get("/status/{player_id}")
def get_player_status(player_id: str):
    if player_id not in player_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Player {player_id} not found."
        )

    times_flagged     = int(player_flag_counts[player_id])
    sessions_observed = int(player_session_counts[player_id])
    is_banned         = times_flagged >= BAN_THRESHOLD

    return {
        "player_id"        : player_id,
        "player_name"      : player_registry[player_id],
        "times_flagged"    : times_flagged,
        "sessions_observed": sessions_observed,
        "lobby"            : "shadow_lobby" if is_banned else "normal_lobby",
        "verdict"          : "BANNED" if is_banned else "ACTIVE"
    }


@app.get("/dashboard")
def get_dashboard():
    total  = len(player_registry)
    banned = sum(
        1 for pid in player_registry
        if player_flag_counts[pid] >= BAN_THRESHOLD
    )
    players = []

    for pid, name in player_registry.items():
        flags    = int(player_flag_counts[pid])
        sessions = int(player_session_counts[pid])
        is_b     = flags >= BAN_THRESHOLD
        players.append({
            "player_id"        : pid,
            "player_name"      : name,
            "times_flagged"    : flags,
            "sessions_observed": sessions,
            "lobby"            : "shadow_lobby" if is_b else "normal_lobby",
            "verdict"          : "BANNED" if is_b else "ACTIVE"
        })

    return {
        "total_players_tracked": total,
        "total_banned"         : banned,
        "total_active"         : total - banned,
        "ban_threshold"        : BAN_THRESHOLD,
        "players"              : sorted(players, key=lambda x: -x["times_flagged"])
    }