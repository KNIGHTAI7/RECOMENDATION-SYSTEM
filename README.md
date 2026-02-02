🎧 BeatBuddy – Music Recommendation System

🔗 Live App:
https://recomendation-system-xwz3ywjnbmj8k8djqnhasm.streamlit.app/

📌 Overview

BeatBuddy is a music recommendation system built using unsupervised machine learning techniques.
It recommends songs based on audio feature similarity using K-Means clustering and cosine similarity.

Users can select a song, and the system recommends musically similar tracks from the same cluster.


🎯 Objective

To build a core music recommender system

Use unsupervised learning (K-Means) instead of user ratings

Recommend songs based on musical features

Deploy the model as an interactive Streamlit web application


🧠 Approach Used

This project uses a content-based recommendation approach:

Songs are represented using audio features

Songs are grouped into clusters using K-Means

Recommendations are generated from the same cluster using cosine similarity


📂 Dataset Description

The dataset is Spotify-like and contains the following key columns:

🎵 Metadata

track_name

artists

popularity

🎚 Audio Features

danceability

energy

acousticness

instrumentalness

liveness

valence

tempo

These features describe the musical characteristics of each track.

⚙️ Project Workflow
1️⃣ Data Loading

Dataset is loaded from dataset.csv or clustered_songs.csv

Missing values and duplicates are removed

2️⃣ Feature Scaling

Numerical audio features are standardized using StandardScaler

3️⃣ Clustering

K-Means clustering groups songs into musical clusters

Number of clusters (K) can be adjusted dynamically from the UI

4️⃣ Similarity Calculation

Cosine similarity is used to measure similarity between songs inside the same cluster

5️⃣ Recommendation Generation

When a user selects a song:

Its cluster is identified

Top similar songs from that cluster are recommended


6️⃣ Deployment

Deployed using Streamlit Cloud

Interactive UI for song selection and recommendations


🖥️ Web App Features

🎵 Song selection dropdown

🎛 Adjustable number of clusters (K)

🔁 Adjustable number of recommendations

📊 Cluster distribution visualization

📥 Download clustered dataset option

⚡ Fast and lightweight interface


🧪 Algorithms Used
Task	Algorithm
Feature Scaling	StandardScaler
Clustering	K-Means
Similarity Measure	Cosine Similarity
Dimensionality Reduction (optional)	PCA
📈 Why Unsupervised Learning?

No explicit user ratings required

Works well with audio feature data

Easily scalable

Suitable for cold-start problems


🧱 Tech Stack

Python

Pandas & NumPy

Scikit-learn

Streamlit

Matplotlib / Seaborn (optional)


👤 Author

Priyanshu
Machine Learning & Data Science Enthusiast
📌 Project built for learning, experimentation, and deployment practice.