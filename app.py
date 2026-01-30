from flask import Flask, render_template, request
import joblib
import os
import gdown
import csv

app = Flask(__name__)

# ================= DOWNLOAD MODEL FROM GOOGLE DRIVE =================

MODEL_URL = "https://drive.google.com/uc?id=1ihaLp2BxbbFu_HbvwcGV0I-4UmkisyfH"

if not os.path.exists("model.pkl"):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, "model.pkl", quiet=False)

# Load model
model = joblib.load("model.pkl")

# ================= LOAD CSV WITHOUT PANDAS =================

known_teams = set()
known_venues = set()

with open("ODI_Match_info.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        known_teams.add(row["team1"])
        known_teams.add(row["team2"])
        known_venues.add(row["venue"])

known_teams = sorted(list(known_teams))
known_venues = sorted(list(known_venues))

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html", teams=known_teams, venues=known_venues)

@app.route("/predict", methods=["POST"])
def predict():
    team1 = request.form["team1"]
    team2 = request.form["team2"]
    venue = request.form["venue"]
    toss_winner = request.form["toss_winner"]
    toss_decision = request.form["toss_decision"]
    season_type = request.form["season_type"]

    # Create input dictionary
    input_data = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "season_type": season_type,
        "dl_applied": 0,
        "team1_strength": 0.5,
        "team2_strength": 0.5,
        "team1_bat_strength": 1.0,
        "team2_bat_strength": 1.0,
        "team1_bowl_strength": 1.0,
        "team2_bowl_strength": 1.0,
        "venue_strength": 0.5
    }

    # sklearn needs list of dict
    input_data = [input_data]

    proba = model.predict_proba(input_data)[0]

    return render_template("result.html",
                           team1=team1,
                           team2=team2,
                           prob1=round(proba[0]*100, 2),
                           prob2=round(proba[1]*100, 2))


# ================= RUN APP =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
