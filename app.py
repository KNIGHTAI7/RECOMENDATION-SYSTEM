import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page setup
st.set_page_config(page_title='BeatBuddy', page_icon='🎧', layout='centered')
st.title('🎧 BeatBuddy — Simple Music Recommender (Core)')

st.markdown("""
### 🧠 Quick Guide:
- Place your `clustered_songs.csv` or `dataset.csv` in the app folder **before deployment**, or upload manually below.  
- Required numeric columns:  
  `danceability`, `energy`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `popularity`  
- The app will cluster tracks and recommend similar songs.
---
""")

# ---------- Dataset Loading Logic ----------
@st.cache_data
def load_data():
    """Load pre-clustered or raw dataset automatically."""
    if os.path.exists("clustered_songs.csv"):
        st.info("✅ Loaded pre-clustered dataset.")
        df = pd.read_csv("clustered_songs.csv")
    elif os.path.exists("dataset.csv"):
        st.warning("⚠️ Loaded raw dataset (clustering will run).")
        df = pd.read_csv("dataset.csv")
    else:
        st.error("❌ No local dataset found. Please upload a CSV below.")
        st.stop()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

@st.cache_data
def load_uploaded(file):
    """Load uploaded CSV."""
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# ---------- File Upload / Local Load ----------
uploaded = st.file_uploader("📁 Upload your dataset (optional)", type=['csv'])

if uploaded is not None:
    df = load_uploaded(uploaded)
    st.success("✅ Loaded dataset from uploaded file.")
else:
    df = load_data()

st.write("### 🔍 Dataset Preview")
st.dataframe(df.head(10))

# ---------- Required Columns Check ----------
required_features = [
    'danceability', 'energy', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'popularity'
]

missing = [col for col in required_features if col not in df.columns]
if missing:
    st.error(f"Missing required numeric columns: {missing}")
    st.stop()

# ---------- Preprocessing ----------
df = df.drop_duplicates().dropna(subset=required_features + ['track_name']).reset_index(drop=True)
features = df[required_features].astype(float)

# ---------- Clustering ----------
scaler = StandardScaler()
X = scaler.fit_transform(features)

st.sidebar.header("⚙️ Settings")

k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=12, value=5, step=1)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
n_rec = st.sidebar.slider("Recommendations per query", min_value=1, max_value=10, value=5)

kmeans = KMeans(n_clusters=k, random_state=int(random_state))
labels = kmeans.fit_predict(X)
df["cluster"] = labels

st.write("### 📊 Cluster Distribution")
st.bar_chart(df["cluster"].value_counts().sort_index())

# ---------- Recommendation Function ----------
def recommend(song_name, n_recommendations=5):
    if song_name not in df["track_name"].values:
        return None
    idx = df.index[df["track_name"] == song_name][0]
    song_cluster = int(df.loc[idx, "cluster"])
    cluster_idx = df[df["cluster"] == song_cluster].index.tolist()

    if len(cluster_idx) <= 1:
        return pd.DataFrame(columns=["track_name", "artists", "similarity"])

    song_vec = X[idx].reshape(1, -1)
    cluster_vecs = X[cluster_idx]
    sims = cosine_similarity(song_vec, cluster_vecs)[0]

    recs = df.loc[cluster_idx].copy()
    recs["similarity"] = sims
    recs = recs.sort_values(by="similarity", ascending=False)
    recs = recs[recs["track_name"] != song_name]

    return recs[["track_name", "artists", "cluster", "similarity"]].head(n_recommendations)

# ---------- UI Interaction ----------
st.write("---")
st.subheader("🎵 Get Song Recommendations")

song_choice = st.selectbox("Select a song:", options=sorted(df["track_name"].unique()))

if st.button("Recommend"):
    recs = recommend(song_choice, n_recommendations=n_rec)
    if recs is None:
        st.error("Song not found in the dataset.")
    elif recs.shape[0] == 0:
        st.info("No other songs in the same cluster to recommend.")
    else:
        st.success(f"🎧 Top {recs.shape[0]} recommendations for **{song_choice}**:")
        st.dataframe(recs.reset_index(drop=True))

        # Download clustered file
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download clustered_songs.csv",
                           data=csv,
                           file_name="clustered_songs.csv",
                           mime="text/csv")

st.write("---")
st.markdown("**ℹ️ Note:** This is the core version of BeatBuddy. Place your `clustered_songs.csv` in the same repo for faster loading.")
