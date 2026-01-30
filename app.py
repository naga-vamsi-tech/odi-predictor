from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import gdown

app = Flask(__name__)

# Google Drive model link
MODEL_URL = "https://drive.google.com/uc?id=1ihaLp2BxbbFu_HbvwcGV0I-4UmkisyfH"

# Download model if not exists
if not os.path.exists("model.pkl"):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, "model.pkl", quiet=False)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load CSV dataset
df = pd.read_csv("ODI_Match_info.csv")

@app.route("/")
def index():
    teams = sorted(df["team1"].unique())
    venues = sorted(df["venue"].unique())
    seasons = sorted(df["season"].unique())
    return render_template("index.html", teams=teams, venues=venues, seasons=seasons)

@app.route("/predict", methods=["POST"])
def predict():
    team1 = request.form["team1"]
    team2 = request.form["team2"]
    venue = request.form["venue"]
    toss_winner = request.form["toss_winner"]
    toss_decision = request.form["toss_decision"]
    season = request.form["season"]

    input_data = pd.DataFrame([[team1, team2, venue, toss_winner, toss_decision, season]],
                              columns=["team1", "team2", "venue", "toss_winner", "toss_decision", "season"])

    proba = model.predict_proba(input_data)[0]
    winner = team1 if proba[0] > proba[1] else team2
    confidence = max(proba) * 100

    return render_template("result.html", winner=winner, confidence=round(confidence, 2),
                           team1=team1, team2=team2, venue=venue, season=season,
                           toss_winner=toss_winner, toss_decision=toss_decision)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
