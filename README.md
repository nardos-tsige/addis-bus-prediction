# Addis Bus Delay Prediction System

## AI Bootcamp - AAU Tech Club | Final Capstone Challenge

---

### Purpose

This project demonstrates a machine learning solution for predicting bus arrival delays in Addis Ababa. The system shows how real-world delay prediction can be implemented using historical data.

**⚠️ IMPORTANT NOTE:** This project uses **synthetic data** for demonstration purposes only. It showcases the complete pipeline and methodology that would be applied to real data. The approach is production-ready and can be deployed with actual bus tracking data.

**Key Objectives:**
- Demonstrate end-to-end ML pipeline for delay prediction
- Identify delay patterns by route, time, weather, and traffic
- Provide a framework that transport authorities can implement with real data

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

### Data Notice

**This project uses synthetic data created for demonstration purposes.** The methodology shown is exactly what would be used with real data. To deploy with real data:
1. Replace the synthetic CSV with actual bus tracking data
2. Ensure columns match the expected format
3. Re-run the training pipeline

---

### Sample Prediction Output (on synthetic data)

| Route | Day & Time | Weather | Traffic | Predicted Delay |
|-------|------------|---------|---------|-----------------|
| Piassa-Megenagna | Saturday 8:00 | Clear | Moderate | 9.9 min |
| Mexico-Piassa | Monday 17:00 | Heavy Rain | Heavy | 27.1 min |
| Bole-Mexico | Friday 22:00 | Light Rain | Severe | 33.3 min |

---

### What This Project Demonstrates

- Complete ML pipeline from data to deployment
- Feature engineering for time, weather, and traffic data
- Model comparison and selection (5 algorithms tested)
- Production-ready prediction API
- Reproducible workflow for real-world application

---

### License

This project is for educational purposes as part of the AI Bootcamp Final Capstone Challenge at AAU Tech Club.

---

