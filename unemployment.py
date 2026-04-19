import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib  
from sqlalchemy import create_engine 
import json
import requests 

@st.cache_data 
def get_data_from_sql() : 
    engine = create_engine('mysql+pymysql://root:20056@localhost/UNEMPLOYEE') 
    query = """
    SELECT * FROM unemployee_db 
    JOIN feature ON unemployee_db.ID = feature.ID_F
    """ 

    df = pd.read_sql(query, con=engine)
    return df 

df = get_data_from_sql() 

st.title("Unemployment Analysis & Prediction Project")

tab1 , tab_map ,tab2 , tab3 = st.tabs(["Overview","Interactive Map", "Insights", "Predictor"]) 
with tab1:
    st.header("Overview") 
    st.write("This project analyzes and predicts unemployment rates in India using machine learning, focusing on regional variations and the impact of time.") 
    
    m1, m2, m3 = st.columns(3) 
    m1.metric("Total Records", len(df)) 
    m2.metric("Avg Unemployment Rate", f"{df['Estimated_Unemployment_Rate'].mean():.2f}%") 
    m3.metric("Max Unemployment Rate", f"{df['Estimated_Unemployment_Rate'].max():.2f}%")
    
    st.divider()
    monthly_trend = df.groupby('recored_date')['Estimated_Unemployment_Rate'].mean().reset_index()

    fig = px.line(
        monthly_trend, 
        x='recored_date', 
        y='Estimated_Unemployment_Rate',
        markers=True,
        title="<b>Monthly Unemployment Rate Trend (2019-2020)</b>",
        labels={'recored_date': 'Timeline', 'Estimated_Unemployment_Rate': 'Avg Rate (%)'}
    )

    fig.update_xaxes(
        dtick="M1", 
        tickformat="%b\n%Y",
        ticklabelmode="period"
    )

    fig.update_layout(
        hovermode="x unified",
        template="plotly_dark", 
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Data Preview")
    st.dataframe(df.head(10)) 

with tab_map:
    st.header("🗺️ Interactive Regional Explorer")
    st.write("Select a state from the map or the dropdown to see localized unemployment trends.")
    with open("Indian_States.txt", "r", encoding="utf-8") as f:
        india_states = json.load(f)
    
    df['Region'] = df['Region'].str.strip()
    map_df = df.groupby('Region')['Estimated_Unemployment_Rate'].mean().reset_index()
    fig_map = px.choropleth(
        map_df, geojson=india_states, featureidkey='properties.NAME_1',
        locations='Region', color='Estimated_Unemployment_Rate',
        color_continuous_scale="Reds", hover_name='Region',
        template="plotly_dark"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    selected_region = st.selectbox("Choose a state to explore:", 
                                options=list(df['Region'].unique()))

    state_df = df[df['Region'] == selected_region]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_state = px.line(state_df.groupby('recored_date')['Estimated_Unemployment_Rate'].mean().reset_index(), 
                            x='recored_date', y='Estimated_Unemployment_Rate', 
                            title=f"Unemployment Trend in {selected_region}", markers=True)
        st.plotly_chart(fig_state, use_container_width=True)
    with col2:
        st.metric(f"Avg Rate ({selected_region})", f"{state_df['Estimated_Unemployment_Rate'].mean():.2f}%")
        st.metric("Max Recorded", f"{state_df['Estimated_Unemployment_Rate'].max():.2f}%")
        fig_pie_state = px.pie(state_df, values='Estimated_Unemployment_Rate', names='Area', 
                            hole=0.4, title="Rural vs Urban")
        st.plotly_chart(fig_pie_state, use_container_width=True)

with tab2: 
    st.header("Insights") 
    st.write("This project provides insights into unemployment rates in India, including regional variations and the impact of time.") 

    df['recored_date'] = pd.to_datetime(df['recored_date'])

    covid_start_date = pd.to_datetime('2020-03-31')

    df['period'] = np.where(df['recored_date'] < covid_start_date, 'Pre-COVID', 'Post-COVID')
    
    st.subheader("Unemployment Rate by Period") 
    st.info("""
    **Observation:** A massive spike is visible between April and June 2020.
    **Reason:** This coincides with the COVID-19 national lockdown in India. 
    """)
    
    period_analysis = df.groupby('period')['Estimated_Unemployment_Rate'].mean().reset_index() 
    fig_period = px.bar(
        period_analysis, 
        x='period', 
        y='Estimated_Unemployment_Rate',
        color='period', 
        text_auto='.2f', 
        title="Avg Unemployment: Pre vs Post COVID"
    )
    st.plotly_chart(fig_period, use_container_width=True) 

    st.subheader("2. Rural vs Urban Analysis") 
    fig_area = px.pie(
        df, 
        values='Estimated_Unemployment_Rate', 
        names='Area', 
        title="Distribution: Rural vs Urban", 
        hole=0.4
    ) 
    st.plotly_chart(fig_area, use_container_width=True) 

    st.subheader("3. Top Impacted Regions") 
    top_regions = df.groupby('Region')['Estimated_Unemployment_Rate'].mean().sort_values(ascending=False).reset_index()
    fig_region = px.bar(
        top_regions, 
        x='Region', 
        y='Estimated_Unemployment_Rate', 
        title="Top Impacted Regions", 
        text_auto='.2f',
        color='Estimated_Unemployment_Rate'
    ) 
    st.plotly_chart(fig_region, use_container_width=True) 

    st.subheader("4. Regional & Area Breakdown") 
    fig_sun = px.sunburst(
        df, 
        path=['Region', 'Area'], 
        values='Estimated_Unemployment_Rate', 
        title="Unemployment Hierarchy"
    )
    st.plotly_chart(fig_sun, use_container_width=True)

    st.success("**Conclusion:** Certain industrial hubs saw a much sharper increase compared to agricultural regions.")

    st.subheader("🔍 Trend Analysis: Smoothing the Noise") 
    fig_trend = go.Figure() 
    
    fig_trend.add_trace(go.Scatter(
        x=df['recored_date'], y=df['Estimated_Unemployment_Rate'], 
        mode='lines', name='Raw Data', 
        line=dict(color='rgba(100, 100, 100, 0.4)', width=1)
    ))

    fig_trend.add_trace(go.Scatter(
        x=df['recored_date'], y=df['rolling_mean'], 
        mode='lines',
        name='Market Trend (Rolling Mean)', 
        line=dict(color='red', width=3)
    )) 

    fig_trend.update_layout(
        title="Actual Unemployment vs. Smoothed Trend",
        hovermode="x unified",
        template="plotly_dark"
    )
    st.plotly_chart(fig_trend, use_container_width=True) 

@st.cache_resource 
def load_models():
    rf = joblib.load('RandomForest.joblib')
    xgb = joblib.load('XGBoost.joblib')
    lr = joblib.load('model_lr.joblib')
    return rf, xgb, lr

models = load_models()
region_map = {val: i for i, val in enumerate(df['Region'].unique())}
area_map = {val: i for i, val in enumerate(df['Area'].unique())}

with tab3:
    st.header("🔍 Unemployment Rate Predictor")
    st.write("Compare different ML models to predict unemployment rates.")

    col1, col2, col3 = st.columns(3)

    with col1:
        region_name = st.selectbox("Select Region", df['Region'].unique())
        year = st.number_input("Year", min_value=2020, max_value=2025, value=2022)
        month = st.slider("Month", 1, 12, 6)

    with col2:
        area_name = st.radio("Area Type", df['Area'].unique())
        lag_val = st.number_input("Previous Month Rate (Lag1)", value=10.0)
        rolling_val = st.number_input("Last 3-Months Average (Rolling)", value=12.0)

    with col3:
        selected_model_name = st.selectbox("Select Model", ["Random Forest", "XGBoost", "Linear Regression"])
    if st.button("Predict Rate 🚀"):
        
        m_sin = np.sin(2 * np.pi * month / 12)
        m_cos = np.cos(2 * np.pi * month / 12)

        region_idx = region_map[region_name]
        area_idx = area_map[area_name]
        
        avg_participation = df['Estimated Labour Participation Rate (%)'].mean() if 'Estimated Labour Participation Rate (%)' in df.columns else 0
        avg_employed = df['Estimated Employed'].mean() if 'Estimated Employed' in df.columns else 0
        
        labour_employed_ratio = avg_employed / avg_participation if avg_participation != 0 else 0
        region_month_interaction = region_idx * month
        relative_unemployment = lag_val / (rolling_val if rolling_val != 0 else 1)

        input_data = {
            'Region_Encoder': [region_idx],
            'Area_Encoder': [area_idx],
            'Estimated Labour Participation Rate (%)': [avg_participation],
            'Estimated Employed': [avg_employed],
            'Month': [month],
            'Year': [year],
            'Period_Encoder': [1 if year >= 2020 and month >= 4 else 0],
            'unemployment_lag1': [lag_val],
            'Rolling_Mean': [rolling_val],
            'diff_unemployee': [rolling_val - lag_val],
            'Labour_Employed_Ratio': [labour_employed_ratio],
            'Region_Month_Interaction': [region_month_interaction],
            'Relative_Unemployment': [relative_unemployment],
            'month_sin': [m_sin],
            'month_cos': [m_cos]
        }

        features_df = pd.DataFrame(input_data)
        try:
            features_df = features_df[models[0].feature_names_in_]
            
            if selected_model_name == "Random Forest":
                prediction = models[0].predict(features_df)
            elif selected_model_name == "XGBoost":
                prediction = models[1].predict(features_df)
            else:
                prediction = models[2].predict(features_df)

            st.markdown("---")
            st.success(f"### Predicted Rate: {prediction[0]:.2f}%")
            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.write("Required features by model:", list(models[0].feature_names_in_))