import os
import gdown
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ================= GOOGLE DRIVE MODEL DOWNLOAD =================

MODEL_URL = "https://drive.google.com/uc?id=1ihaLp2BxbbFu_HbvwcGV0I-4UmkisyfH"

# Download model if not exists
if not os.path.exists("model.pkl"):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, "model.pkl", quiet=False)

# Load model
model = joblib.load("model.pkl")

# ================= LOAD DATA =================

match_info = pd.read_csv("ODI_Match_info.csv")
teams = sorted(set(match_info['team1']).union(set(match_info['team2'])))
venues = sorted(match_info['venue'].dropna().unique())

# ================= HOME PAGE =================

@app.route("/")
def home():
    return render_template("index.html", teams=teams, venues=venues)

# ================= PREDICT =================

@app.route("/predict", methods=["POST"])
def predict():
    team1 = request.form.get("team1")
    team2 = request.form.get("team2")
    venue = request.form.get("venue")
    season = request.form.get("season")
    toss_winner = request.form.get("toss_winner")
    toss_decision = request.form.get("toss_decision")

    # Same team error
    if team1 == team2:
        return "<h2 style='color:red;'>❌ Team 1 and Team 2 cannot be same!</h2>"

    # Create input data
    input_data = pd.DataFrame({
        'team1':[team1],
        'team2':[team2],
        'venue':[venue],
        'toss_winner':[toss_winner],
        'toss_decision':[toss_decision],
        'season_type':[season],
        'dl_applied':[0],
        'team1_strength':[0.5],
        'team2_strength':[0.5],
        'team1_bat_strength':[1],
        'team2_bat_strength':[1],
        'team1_bowl_strength':[1],
        'team2_bowl_strength':[1],
        'venue_strength':[0.5]
    })

    # Predict
    proba = model.predict_proba(input_data)[0]
    classes = model.classes_

    team_probs = {}
    for t, p in zip(classes, proba):
        if t == team1 or t == team2:
            team_probs[t] = p

    winner = max(team_probs, key=team_probs.get)
    prob = round(team_probs[winner] * 100, 2)

    return render_template("result.html",
                           winner=winner,
                           prob=prob,
                           team1=team1,
                           team2=team2,
                           venue=venue,
                           season=season,
                           toss_winner=toss_winner,
                           toss_decision=toss_decision)

# ================= RUN FOR RENDER =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
