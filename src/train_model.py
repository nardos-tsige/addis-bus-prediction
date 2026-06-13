import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

print("="*70)
print("BUS DELAY PREDICTION - MODEL TRAINING")
print("="*70)

df = pd.read_csv('data/synthetic_data_2026-06-13.csv')
print(f"Loaded {len(df):,} records")

print("\nEngineering features...")
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
target = 'total_delay_minutes'

X = df[numeric_features + categorical_features]
y = df[target]

preprocessor = ColumnTransformer([
    ('scaler', StandardScaler(), numeric_features),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")

print("\nTraining Random Forest...")
pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} min")
print(f"RMSE: {rmse:.2f} min")
print(f"R2: {r2:.4f}")

print("\n" + "="*70)
print(f"BEST MODEL: Random Forest")
print(f"MAE: {mae:.2f} minutes")
print(f"R2: {r2:.4f}")
print("="*70)

joblib.dump(pipeline, 'models/best_model.pkl')
print("\nModel saved: models/best_model.pkl")
print("Training complete!")