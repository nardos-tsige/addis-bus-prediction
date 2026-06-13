import pandas as pd

def validate_and_load(filepath='data/synthetic_data_2026-06-13.csv'):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records")
    print(f"Columns: {df.columns.tolist()}")
    
    expected = ['timestamp', 'bus_id', 'route', 'passenger_count', 'day_of_week',
                'hour_of_day', 'month', 'day', 'weather_condition', 'distance_km',
                'special_event', 'is_holiday', 'total_delay_minutes',
                'scheduled_arrival_time', 'actual_arrival_time', 'traffic_intensity']
    
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
    else:
        print("Data structure validated!")
    
    return df

if __name__ == "__main__":
    df = validate_and_load()
    print(f"Data ready for training!")