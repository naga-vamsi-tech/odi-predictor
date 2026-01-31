from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import requests

app = Flask(__name__)

# -------------------------------------------------
# LOAD DATASET (for dropdowns)
# -------------------------------------------------
df = pd.read_csv("ODI_Match_info.csv")

# -------------------------------------------------
# DOWNLOAD & LOAD MODEL (from Hugging Face)
# -------------------------------------------------
MODEL_URL = "https://huggingface.co/vamsi-30/odi-predictor-model/resolve/main/model(1).pkl"

if not os.path.exists("model.pkl"):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open("model.pkl", "wb") as f:
        f.write(r.content)

model = joblib.load("model.pkl")

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
@app.route("/")
def index():
    teams = sorted(set(df["team1"]).union(set(df["team2"])))
    venues = sorted(df["venue"].dropna().unique())
    seasons = ["Summer", "Rainy", "Winter"]

    return render_template(
        "index.html",
        teams=teams,
        venues=venues,
        seasons=seasons
    )

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    team1 = request.form["team1"]
    team2 = request.form["team2"]
    venue = request.form["venue"]
    toss_winner = request.form["toss_winner"]
    toss_decision = request.form["toss_decision"]
    season = request.form["season"]

    # Validation
    if team1 == team2:
        return "Error: Team 1 and Team 2 cannot be the same"

    # Prepare input for ML model
    input_data = pd.DataFrame([{
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "season": season
    }])

    # Predict probabilities
    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    prob_dict = dict(zip(classes, probs))

    team1_prob = round(prob_dict.get(team1, 0) * 100, 2)
    team2_prob = round(prob_dict.get(team2, 0) * 100, 2)

    # Decide winner & loser
    if team1_prob >= team2_prob:
        winner = team1
        loser = team2
        winner_prob = team1_prob
        loser_prob = team2_prob
    else:
        winner = team2
        loser = team1
        winner_prob = team2_prob
        loser_prob = team1_prob

    # -----------------------------
    # SIMPLE EXPLANATION (for sir)
    # -----------------------------
    reasons = []

    if toss_winner == winner:
        reasons.append("won the toss")

    if toss_decision == "bat":
        reasons.append("chose to bat first")

    reasons.append("better historical performance in similar conditions")

    explanation = ", ".join(reasons)

    return render_template(
        "result.html",
        winner=winner,
        winner_prob=winner_prob,
        loser=loser,
        loser_prob=loser_prob,
        explanation=explanation,
        team1=team1,
        team2=team2,
        venue=venue,
        season=season,
        toss_winner=toss_winner,
        toss_decision=toss_decision
    )

# -------------------------------------------------
# RUN APP
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
