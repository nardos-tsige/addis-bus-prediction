# Addis Bus Delay Prediction System

## AI Bootcamp - AAU Tech Club | Final Capstone Challenge

---

### Purpose

This project predicts bus arrival delays in Addis Ababa using machine learning. The system helps commuters plan their trips and enables transport authorities to make data-driven decisions for route optimization and resource allocation.

**Key Objectives:**
- Predict delay times with high accuracy (1.73 minutes MAE)
- Identify delay patterns by route, time, weather, and traffic
- Provide actionable insights for better urban mobility

---

### Tools & Technologies

| Category | Tools Used |
|----------|------------|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest, Gradient Boosting, Ridge, Lasso, Linear Regression) |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Environment | Jupyter Notebook, VS Code |

**Best Model:** Random Forest Regressor (MAE: 1.73 min, R²: 0.9301)

---

### Key Features Engineered

- is_weekend | is_rush_hour | is_night
- weather_severity | traffic_numeric | has_special_event

---

### Sample Prediction Output

| Route | Day & Time | Weather | Traffic | Predicted Delay |
|-------|------------|---------|---------|-----------------|
| Piassa-Megenagna | Saturday 8:00 | Clear | Moderate | 9.9 min |
| Mexico-Piassa | Monday 17:00 | Heavy Rain | Heavy | 27.1 min |
| Bole-Mexico | Friday 22:00 | Light Rain | Severe | 33.3 min |

---

### License

This project is for educational purposes as part of the AI Bootcamp Final Capstone Challenge at AAU Tech Club.

---

