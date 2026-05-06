import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. DATA ACQUISITION (Simulating Spotipy API Output)
# In a real scenario, you would use: sp.search(q='year:2024', type='track')
data = {
    'song_name': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E'],
    'energy': [0.8, 0.4, 0.9, 0.2, 0.7],
    'danceability': [0.75, 0.5, 0.85, 0.3, 0.6],
    'tempo': [120, 90, 128, 70, 115],
    'loudness': [-5, -12, -4, -18, -6],
    'is_popular': [1, 0, 1, 0, 1]  # Target Variable
}

df = pd.DataFrame(data)

# 2. DATA PREPROCESSING (SQL-like cleaning in Python)
def clean_spotify_data(df):
    # Handling missing values and normalizing features
    df = df.dropna()
    return df

df_cleaned = clean_spotify_data(df)

# 3. FEATURE SELECTION
X = df_cleaned[['energy', 'danceability', 'tempo', 'loudness']]
y = df_cleaned['is_popular']

# 4. PREDICTIVE MODELING (Machine Learning)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Random Forest as it's robust for feature importance
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. EVALUATION
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100}%")
print(classification_report(y_test, predictions))

# 6. EXPORT FOR VISUALIZATION (Power BI/Tableau)
# This mimics your workflow of moving data from Python to BI tools
df_cleaned.to_csv('spotify_trends_for_bi.csv', index=False)
