from flask import Flask, render_template, request
import pandas as pd
import os
import joblib
import requests

app = Flask(__name__)

# ===============================
# DOWNLOAD MODEL FROM HUGGING FACE
# ===============================
MODEL_URL = "https://huggingface.co/vamsi-30/odi-predictor-model/resolve/main/model(1).pkl"
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading ML model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load ML model
model = joblib.load(MODEL_PATH)
print("Model Loaded Successfully!")

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("ODI_Match_info.csv")

known_teams = sorted(set(df["team1"]).union(set(df["team2"])))
known_venues = sorted(df["venue"].dropna().unique())

# ===============================
# HOME PAGE
# ===============================
@app.route("/")
def index():
    seasons = ["Summer", "Rainy", "Winter"]
    return render_template("index.html", teams=known_teams, venues=known_venues, seasons=seasons)

# ===============================
# PREDICTION ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    team1 = request.form["team1"]
    team2 = request.form["team2"]
    venue = request.form["venue"]
    toss_winner = request.form["toss_winner"]
    toss_decision = request.form["toss_decision"]
    season = request.form["season"]

    # -------- VALIDATION --------
    if team1 == team2:
        return "❌ ERROR: Team 1 and Team 2 cannot be the same!"

    if team1 not in known_teams or team2 not in known_teams:
        return "❌ ERROR: One or both teams are not in dataset!"

    # -------- CREATE INPUT DATAFRAME --------
    input_data = pd.DataFrame({
        "team1": [team1],
        "team2": [team2],
        "venue": [venue],
        "toss_winner": [toss_winner],
        "toss_decision": [toss_decision],
        "season_type": [season],
        "dl_applied": [0],
        "team1_strength": [0.5],
        "team2_strength": [0.5],
        "team1_bat_strength": [1.0],
        "team2_bat_strength": [1.0],
        "team1_bowl_strength": [1.0],
        "team2_bowl_strength": [1.0],
        "venue_strength": [0.5]
    })

    # -------- PREDICT --------
    proba = model.predict_proba(input_data)[0]
    classes = model.classes_

    team_probs = {team: p for team, p in zip(classes, proba) if team in [team1, team2]}
    winner = max(team_probs, key=team_probs.get)
    confidence = team_probs[winner] * 100

    return render_template(
        "result.html",
        winner=winner,
        confidence=round(confidence, 2),
        team1=team1,
        team2=team2,
        venue=venue,
        season=season,
        toss_winner=toss_winner,
        toss_decision=toss_decision
    )

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
