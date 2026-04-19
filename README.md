# 📈 India Unemployment Analysis & Prediction Dashboard

An end-to-end Data Science project that analyzes unemployment trends across India (2020-2022) and predicts future rates using optimized Machine Learning models.

## 🚀 Key Features
- **Interactive Geospatial Mapping:** Heatmaps showing unemployment density across Indian states using GeoJSON.
- **COVID-19 Impact Analysis:** Comparative visualizations of labor market shifts before and after March 2020.
- **Multi-Model Predictor:** User-friendly interface to compare predictions from **Random Forest**, **XGBoost**, and **Linear Regression**.
- **Dynamic Feature Engineering:** Includes automated lag features, rolling means, and cyclical time encoding.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Database:** SQL
- **Machine Learning:** Scikit-Learn, XGBoost, Joblib
- **Visualization:** Plotly Express

## 📁 Project Structure
- `unemployment.py`: The main application script.
- `Indian_States.txt`: Local GeoJSON file for map rendering.
- `RandomForest.joblib` / `XGBoost.joblib`: Pre-trained model files.
- `data_fetch.py`: SQL connection and data extraction logic.

## 🔧 How to Run
1. Clone the repository.
2. Ensure you have the required libraries: `pip install streamlit pandas plotly xgboost scikit-learn`.
3. Run the command: `streamlit run unemployment.py`.
