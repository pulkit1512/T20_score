import streamlit as st
import pickle
import pandas as pd
import numpy as np


# --- 1. PERFORMANCE METRICS (From Notebook) ---
model_performance = {
    "XGBoost": {"rmse": 8.52, "file": 'xgb_model_t20_score.pkl'},
    "LightGBM": {"rmse": 8.44, "file": 'lgbm_model_t20_score.pkl'},
    "Gradient Boosting": {"rmse": 12.28, "file": 'gb_model_t20_score.pkl'},
    "Linear Regression": {"rmse": 22.71, "file": 'lr_model_t20_score.pkl'}
}

# 2026 T20 World Cup Teams Colors
TEAM_COLORS = {
    'India': '#1D4ED8', 'Sri Lanka': '#002060', 'Afghanistan': '#0058A8',
    'Australia': '#FFCD00', 'Bangladesh': '#006A4E', 'England': '#E61B2E',
    'South Africa': '#007A33', 'USA': '#002868', 'West Indies': '#7B0031',
    'Ireland': '#169B62', 'New Zealand': '#000000', 'Pakistan': '#01411C',
    'Netherlands': '#F36C21', 'Italy': '#008C45', 'Namibia': '#003580',
    'Zimbabwe': '#E4002B', 'Nepal': '#DC143C', 'Oman': '#DA291C',
    'Canada': '#FF0000', 'United Arab Emirates': '#00732F'
}

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_all_assets():
    # Ensure these filenames match your local directory
    teams = pickle.load(open('teams_t20_score.pkl', 'rb'))
    cities = pickle.load(open('cities_t20_score.pkl', 'rb'))
    models = {name: pickle.load(open(info['file'], 'rb')) for name, info in model_performance.items()}
    return teams, cities, models

teams, cities, loaded_models = load_all_assets()

# --- 3. SIDEBAR NAVIGATION & MATCH SETUP ---
st.sidebar.title("🎮 T20WC Dashboard")
st.sidebar.divider()
st.sidebar.header("Match Setup")
batting_team = st.sidebar.selectbox('Batting Team', sorted(teams))
bowling_team = st.sidebar.selectbox('Bowling Team', sorted(teams))
city = st.sidebar.selectbox('Venue City', sorted(cities))
algorithm = st.sidebar.selectbox("Prediction Algorithm", list(model_performance.keys()))

# Set Dynamic Colors for Alignment
color1 = TEAM_COLORS.get(batting_team, '#1E3A8A')
color2 = TEAM_COLORS.get(bowling_team, '#1E293B')

# --- 4. PERFECTLY ALIGNED DYNAMIC CSS ---
st.markdown(f"""
    <style>
    /* Main Background Alignment */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, {color1} 0%, #0F172A 50%, {color2} 100%) !important;
        color: #ffffff;
    }}

    /* Sidebar Background Alignment - Perfectly Blended */
    [data-testid="stSidebarContent"] {{
        background: linear-gradient(180deg, {color1} 0%, {color2} 100%) !important;
        border-right: 2px solid rgba(255, 255, 255, 0.2);
    }}

    /* Sidebar Text/Labels */
    [data-testid="stSidebar"] * {{
        color: white !important;
        font-weight: bold;
    }}
    
    /* Metric styling */
    div[data-testid="stMetric"] {{ 
        background: rgba(255, 255, 255, 0.1); 
        border: 1px solid rgba(255, 255, 255, 0.2); 
        border-radius: 10px; 
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
st.title(f"🏏 T20 World Cup Score Predictor")

col_input, col_trend = st.columns([2, 1])

with col_input:
    st.subheader("📝 Live Match Data")
    sc1, sc2, sc3 = st.columns(3)
    score = sc1.number_input('Current Score', min_value=0, step=1, value=50)
    wickets = sc2.number_input('Total Wickets Down', min_value=0, max_value=9, step=1, value=2)
    overs = sc3.number_input('Overs Completed', min_value=0.0, max_value=19.5, step=0.1, value=10.0)

with col_trend:
    st.subheader("📈 Momentum")
    l5_runs = st.number_input('Runs in Last 5 Overs', min_value=0, step=1, value=30)
    l5_wickets = st.number_input('Wickets in Last 5 Overs', min_value=0, max_value=9, step=1, value=1)

# --- 6. PREDICTION LOGIC WITH EXCEPTION HANDLING ---
if st.button('🔥 RUN PREDICTION'):
    # Validation Checks
    if batting_team == bowling_team:
        st.error(f"❌ Selection Error: Batting and Bowling teams must be different. (Current: {batting_team})")
    
    elif overs <= 5.0:
        st.error("❌ Data Error: Please enter more than 5.0 overs for an accurate prediction.")
        
    elif l5_wickets > wickets:
        st.error(f"❌ Logic Error: Wickets in last 5 overs ({l5_wickets}) cannot be greater than total wickets ({wickets}).")
        
    elif l5_runs > score:
        st.error(f"❌ Logic Error: Runs in last 5 overs ({l5_runs}) cannot exceed total score ({score}).")
    
    else:
        try:
            # 1. Feature Engineering
            balls_left = 120 - (int(overs) * 6 + int((overs % 1) * 10))
            wickets_left = 10 - wickets
            balls_bowled = 120 - balls_left
            crr = score / (balls_bowled / 6)
            
            # 2. Aggression Bonus Calculation
            agg_bonus = 0
            if wickets_left >= 3 and overs >= 7:
                agg_bonus = (wickets_left * 1.4) + (crr - 8.9) * 1.70 + (balls_left * 0.05) - (l5_wickets * 2.0)+(l5_runs*0.2)

            # 3. Create DataFrame with correct column names for the pipeline
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'current_score': [score],
                'balls_left': [balls_left],
                'wickets_left': [wickets_left],
                'crr': [crr],
                'last_five_runs': [l5_runs],
                'last_five_wickets': [l5_wickets]
            })

            # 4. Predict
            pipe = loaded_models[algorithm]
            raw_pred = pipe.predict(input_df)[0]
            
            # 5. Final Calculation
            final_score = int(max(raw_pred + max(0, agg_bonus), score))
            rmse = model_performance[algorithm]['rmse']

            # 6. Display Output
            st.divider()
            st.markdown(f"<h1 style='text-align: center; color: #00ff00;'>Predicted Score: {final_score}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Range (±RMSE): {final_score-int(rmse)} — {final_score+int(rmse)}</h3>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Internal Prediction Error: {e}")
            st.info("Ensure your model pipeline expects columns: batting_team, bowling_team, city, current_score, balls_left, wickets_left, crr, last_five_runs, last_five_wickets.")