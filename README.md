# BeatBuddy — Minimal Streamlit Music Recommender

This repo contains a minimal Streamlit app (app.py) that clusters songs using K-Means on audio features and returns recommendations from the same cluster.

## How to use (locally)
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Put your dataset.csv in the folder (or upload it via the app UI).
3. Run:
```bash
streamlit run app.py
```
4. Open the URL shown in the terminal (usually http://localhost:8501)

## How to deploy to Streamlit Cloud
1. Create a new GitHub repository and push these files:
   - app.py
   - requirements.txt
   - dataset.csv (optional, you can let users upload instead)
   - README.md
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", choose the repo and branch, and deploy.
4. Your app will be live.

## Dataset expectations
- CSV with columns: track_name (string), artists (string), and numeric audio features:
  danceability, energy, acousticness, instrumentalness, liveness, valence, tempo, popularity
