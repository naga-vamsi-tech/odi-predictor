from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import requests

app = Flask(__name__)

# Hugging Face model URL
MODEL_URL = "https://huggingface.co/vamsi-30/odi-predictor-model/resolve/main/model(1).pkl"

# Download model if not exists
if not os.path.exists("model.pkl"):
    print("Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open("model.pkl", "wb") as f:
        f.write(r.content)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset
df = pd.read_csv("ODI_Match_info.csv")

@app.route("/")
def index():
    teams = sorted(df["team1"].unique())
    venues = sorted(df["venue"].unique())
    seasons = sorted(df["season"].unique())
    return render_template("index.html", teams=teams, venues=venues, seasons=seasons)

@app.route("/predict", methods=["POST"])
def predict():
    try:
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

        return render_template("result.html", winner=winner, confidence=round(confidence, 2))

    except Exception as e:
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
