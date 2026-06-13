import pandas as pd
import joblib

print("Loading model...")
pipeline = joblib.load('models/best_model.pkl')
print("Model loaded successfully!")

weather_map = {'Clear': 0, 'Light Rain': 1, 'Heavy Rain': 2}
traffic_map = {'Light': 0, 'Moderate': 1, 'Heavy': 2, 'Severe': 3}

def predict_delay(route, hour, day_of_week, passengers, weather, distance, traffic, bus_id='BUS_0000'):
    df = pd.DataFrame([{
        'bus_id': bus_id,
        'route': route,
        'passenger_count': passengers,
        'day_of_week': day_of_week,
        'hour_of_day': hour,
        'weather_condition': weather,
        'distance_km': distance,
        'special_event': 'None',
        'is_holiday': 0,
        'traffic_intensity': traffic
    }])
    
    df['is_holiday'] = 0
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['is_rush_hour'] = df['hour_of_day'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
    df['is_night'] = df['hour_of_day'].apply(lambda x: 1 if (x >= 23 or x <= 5) else 0)
    df['weather_severity'] = df['weather_condition'].map(weather_map)
    df['has_special_event'] = 0
    df['traffic_numeric'] = df['traffic_intensity'].map(traffic_map)
    
    numeric_features = ['passenger_count', 'hour_of_day', 'distance_km', 'weather_severity', 
                        'traffic_numeric', 'is_weekend', 'is_rush_hour', 'is_night', 
                        'is_holiday', 'has_special_event']
    categorical_features = ['route', 'weather_condition', 'bus_id']
    
    prediction = pipeline.predict(df[numeric_features + categorical_features])[0]
    return prediction

print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

result1 = predict_delay('Piassa-Megenagna', 8, 'Saturday', 45, 'Clear', 15, 'Moderate')
print(f"\nPiassa-Megenagna, Saturday 8:00, Clear, Moderate traffic")
print(f"Predicted delay: {result1:.1f} minutes")

result2 = predict_delay('Mexico-Piassa', 17, 'Monday', 70, 'Heavy Rain', 8, 'Heavy')
print(f"\nMexico-Piassa, Monday 17:00, Heavy Rain, Heavy traffic")
print(f"Predicted delay: {result2:.1f} minutes")

result3 = predict_delay('Bole-Mexico', 22, 'Friday', 25, 'Light Rain', 10, 'Severe')
print(f"\nBole-Mexico, Friday 22:00, Light Rain, Severe traffic")
print(f"Predicted delay: {result3:.1f} minutes")

result4 = predict_delay('Megenagna-Mexico', 6, 'Tuesday', 60, 'Clear', 12, 'Light')
print(f"\nMegenagna-Mexico, Tuesday 6:00, Clear, Light traffic")
print(f"Predicted delay: {result4:.1f} minutes")

print("\n" + "="*60)
print("Ready! Use predict_delay() function for new predictions")
print("="*60)