import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("Loading model and data...")
pipeline = joblib.load('models/best_model.pkl')
df = pd.read_csv('data/synthetic_data_2026-06-13.csv')

print("Engineering features...")
df['is_holiday'] = df['is_holiday'].astype(int)
df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
df['is_rush_hour'] = df['hour_of_day'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
df['is_night'] = df['hour_of_day'].apply(lambda x: 1 if (x >= 23 or x <= 5) else 0)

weather_map = {'Clear': 0, 'Light Rain': 1, 'Heavy Rain': 2}
df['weather_severity'] = df['weather_condition'].map(weather_map)
df['has_special_event'] = (df['special_event'] != 'None').astype(int)

traffic_map = {'Light': 0, 'Moderate': 1, 'Heavy': 2, 'Severe': 3}
df['traffic_numeric'] = df['traffic_intensity'].map(traffic_map)

numeric_features = ['passenger_count', 'hour_of_day', 'distance_km', 'weather_severity', 
                    'traffic_numeric', 'is_weekend', 'is_rush_hour', 'is_night', 
                    'is_holiday', 'has_special_event']
categorical_features = ['route', 'weather_condition', 'bus_id']

preprocessor = ColumnTransformer([
    ('scaler', StandardScaler(), numeric_features),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

preprocessor.fit(df[numeric_features + categorical_features])

encoded_features = preprocessor.named_transformers_['encoder'].get_feature_names_out(categorical_features)
all_features = list(numeric_features) + list(encoded_features)

importances = pipeline.named_steps['regressor'].feature_importances_
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(12, 8))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 15))
bars = plt.barh(range(15), importances[indices][::-1], color=colors[::-1])
plt.yticks(range(15), [all_features[i] for i in indices[::-1]])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 15 Most Important Features for Delay Prediction', fontsize=14)
plt.gca().invert_yaxis()

for i, v in enumerate(importances[indices][::-1]):
    plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("Chart saved as 'feature_importance.png'")