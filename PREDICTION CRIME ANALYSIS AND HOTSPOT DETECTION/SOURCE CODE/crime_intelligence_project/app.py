import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import joblib
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap, MarkerCluster
import pydeck as pdk

# Page Configuration
st.set_page_config(
    page_title="Crime Intelligence Command Center",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Path Resolution for Model Loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# Custom Cyber-Security Command Center CSS Theme
st.markdown(
    """
    <style>
    /* Dark Sci-Fi HUD CSS overrides */
    .stApp {
        background-color: #030712;
        color: #e5e7eb;
        font-family: 'Share Tech Mono', 'Courier New', monospace;
    }
    
    /* Neon Text and Glowing borders */
    h1, h2, h3 {
        color: #00f2fe !important;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.4);
    }
    
    /* Command Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #1e293b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #00f2fe;
    }
    
    .metric-card-alert {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #1e293b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #f43f5e;
    }
    
    .metric-val {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Blinking LIVE Indicator */
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.2; }
        100% { opacity: 1; }
    }
    
    .live-indicator {
        background-color: #f43f5e;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 10px #f43f5e;
        animation: blink 1.5s infinite;
    }
    
    .sys-status {
        font-size: 0.9rem;
        color: #10b981;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 0.05em;
    }
    
    /* Form Containers */
    .stSelectbox, .stSlider {
        margin-bottom: 10px;
    }
    
    /* Custom Sidebar Header */
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: bold;
        color: #00f2fe;
        border-bottom: 2px solid #1e293b;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    
    /* Glassmorphism containers */
    .glass-panel {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
    }
    
    /* Accent glowing borders */
    .glow-cyan { border-top: 3px solid #00f2fe; }
    .glow-pink { border-top: 3px solid #f43f5e; }
    .glow-orange { border-top: 3px solid #f97316; }
    .glow-green { border-top: 3px solid #10b981; }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Load Dataset & Models
@st.cache_data
def load_crime_dataset():
    path = get_path("dataset.csv")
    if os.path.exists(path):
        data = pd.read_csv(path)
        data['date'] = pd.to_datetime(data['date'])
        data['hour'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.hour
        return data
    else:
        # Fallback empty dataframe if dataset doesn't exist
        st.warning("dataset.csv not found. Re-run notebook to generate it.")
        return pd.DataFrame()

@st.cache_resource
def load_ml_models():
    try:
        classifier = joblib.load(get_path("crime_classifier.joblib"))
        forecast_model = joblib.load(get_path("crime_forecast_model.joblib"))
        cluster_model_tuple = joblib.load(get_path("crime_cluster_model.joblib"))
        label_encoders = joblib.load(get_path("label_encoders.joblib"))
        feature_columns = joblib.load(get_path("feature_columns.joblib"))
        scaler = joblib.load(get_path("scaler.joblib"))
        return classifier, forecast_model, cluster_model_tuple, label_encoders, feature_columns, scaler
    except Exception as e:
        st.error(f"Failed to load ML joblib models: {e}")
        return None

# Load resources
df = load_crime_dataset()
models = load_ml_models()

# Validate presence of data and models
if df.empty or models is None:
    st.error("🚨 Critical Error: Project datasets and model files missing. Execute notebook.ipynb first.")
    st.stop()

# Unpack models
hotspot_classifier, risk_model, (kmeans, risk_mapping), label_encoders, feature_columns, scaler = models

# Sidebar Panel
st.sidebar.markdown("<div class='sidebar-header'>⚡ CONTROL TELEMETRY</div>", unsafe_allow_html=True)
st.sidebar.write("🛰️ **COMMUNICATIONS: ONLINE**")
st.sidebar.write("🔒 **SECURE ENCRYPTION: AES-256**")

# System Status Clock
time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
st.sidebar.code(f"SYS_CLOCK: {time_str}")

# Filter Controls
st.sidebar.markdown("### 🔍 SECTOR SELECTOR")
cities = sorted(df['city'].unique())
selected_city = st.sidebar.selectbox("Select City Jurisdiction", cities, index=0)

# Filter dataset by city
city_df = df[df['city'] == selected_city].copy()

# Select Area (filtered based on city)
areas = sorted(city_df['area'].unique())
selected_area = st.sidebar.selectbox("Select Sector Area", areas)

# Select Crime Type
crime_types = sorted(df['crime_type'].unique())
selected_crime_type = st.sidebar.selectbox("Select Threat Vector Type", crime_types)

# Map Theme Selector
st.sidebar.markdown("### 🎨 MAP THEME")
map_theme = st.sidebar.selectbox(
    "Select Map Style",
    ["OpenStreetMap (Highly Readable)", "CartoDB Positron (Light Theme)", "CartoDB Dark Matter (Dark Theme)"],
    index=0
)
tile_mapping = {
    "OpenStreetMap (Highly Readable)": "OpenStreetMap",
    "CartoDB Positron (Light Theme)": "CartoDB positron",
    "CartoDB Dark Matter (Dark Theme)": "CartoDB dark_matter"
}
selected_tile = tile_mapping[map_theme]

# Main Banner Header
col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1:
    st.markdown("<h1>🚨 Crime Intelligence Command Center</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280; font-size:1.1rem; margin-top:-10px;'>Spatial-Temporal Crime Prediction & Threat Detection Telemetry Console</p>", unsafe_allow_html=True)
with col_header_2:
    st.markdown(
        """
        <div style='text-align: right; padding-top: 15px;'>
            <span class='live-indicator'></span><span class='sys-status'>LIVE TELEMETRY STREAMING</span><br/>
            <span style='font-size:0.75rem; color:#6b7280;'>SYSSEC V2.4 // SAMSKRUTI COLLEGE OF ENGINEERING</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# High Level Telemetry Metrics Row
active_hotspots_count = df[df['hotspot_label'] == 1].groupby(['city', 'area']).size().shape[0]
city_hotspots = city_df[city_df['hotspot_label'] == 1].groupby('area').size().shape[0]
avg_arrest_rate = (df[df['arrest_made'] == 'Yes'].shape[0] / df.shape[0]) * 100

col_metric_1, col_metric_2, col_metric_3, col_metric_4 = st.columns(4)
with col_metric_1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val">{len(df):,}</div>
            <div class="metric-lbl">Total Logged Crimes (India)</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_metric_2:
    st.markdown(
        f"""
        <div class="metric-card-alert">
            <div class="metric-val">{active_hotspots_count}</div>
            <div class="metric-lbl">Active Hotspots (Nationwide)</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_metric_3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val">{city_hotspots}</div>
            <div class="metric-lbl">Active Hotspots in {selected_city}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_metric_4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val">{avg_arrest_rate:.1f}%</div>
            <div class="metric-lbl">Arrest / Resolution Rate</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# Navigation Tabs
tab_map, tab_hotspot, tab_forecast, tab_feed, tab_ai = st.tabs([
    "📟 LIVE THREAT GRID MAP", 
    "🎯 HOTSPOT DETECTION GATEWAY", 
    "📈 TEMPORAL RISK FORECAST CONSOLE",
    "📰 INTELLIGENCE INSIGHTS FEED", 
    "🧠 AI ACADEMIC SYSTEM EXPLAINER"
])

# ==================== TAB 1: LIVE MAPS ====================
with tab_map:
    st.markdown("### 🗺️ Real-Time Sector GIS Threat Overlay")
    st.write("Visualizing crime incident distribution, hotspot kernels, and risk clustering outputs from K-Means.")

    map_view = st.radio("Map View Engine Mode", ["Interactive GIS Map (Folium Heatmap)", "3D Spatial Hexagon Density (PyDeck)"], horizontal=True)

    if map_view == "Interactive GIS Map (Folium Heatmap)":
        # Folium mapping
        st.write("🔴 *Red dots indicate High Risk cluster centers, orange denotes Medium Risk, and green denotes Low Risk.*")
        
        # Center coordinates
        city_center_lat = city_df['latitude'].mean()
        city_center_lon = city_df['longitude'].mean()
        
        m = folium.Map(location=[city_center_lat, city_center_lon], zoom_start=12, tiles=selected_tile)
        
        # Add heatmap layer
        heat_data = city_df[['latitude', 'longitude']].values.tolist()
        HeatMap(heat_data, radius=15, blur=10, gradient={0.4: 'blue', 0.65: 'lime', 1.0: 'red'}).add_to(m)
        
        # Determine Area centers coordinates & risk zone
        area_summary = city_df.groupby('area')[['latitude', 'longitude']].mean().reset_index()
        
        # Map clusters for area center
        color_map = {
            "High Risk Zone": "#ff0055",
            "Medium Risk Zone": "#ff9900",
            "Low Risk Zone": "#00ff66"
        }
        
        # Draw clusters
        for idx, row in area_summary.iterrows():
            area_name = row['area']
            # Scale coordinates and predict cluster to map risk zone
            scaled = scaler.transform([[row['latitude'], row['longitude']]])
            cid = kmeans.predict(scaled)[0]
            rzone = risk_mapping[cid]
            
            # Hotspot status
            hlabel = city_df[city_df['area'] == area_name]['hotspot_label'].iloc[0]
            hotspot_text = "🚨 HOTSPOT DECLARED" if hlabel == 1 else "✅ Sector Low Danger"
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=10,
                color=color_map[rzone],
                fill=True,
                fill_color=color_map[rzone],
                fill_opacity=0.8,
                tooltip=f"<b>Sector:</b> {area_name}<br><b>Risk Zone:</b> {rzone}<br><b>Status:</b> {hotspot_text}"
            ).add_to(m)
            
        # Draw a sample of actual crime pins using MarkerCluster for performance
        marker_cluster = MarkerCluster(name="Recent Crime Incidents").add_to(m)
        sample_df = city_df.sample(min(len(city_df), 300), random_state=42)
        for idx, row in sample_df.iterrows():
            # Severity color
            sev_color = "red" if row['severity_level'] == "High" else ("orange" if row['severity_level'] == "Medium" else "blue")
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color=sev_color,
                fill=True,
                fill_color=sev_color,
                fill_opacity=0.6,
                popup=f"ID: {row['crime_id']}<br>Type: {row['crime_type']}<br>Severity: {row['severity_level']}<br>Time: {row['time']}"
            ).add_to(marker_cluster)

        # Draw map
        folium_static(m, width=1100, height=550)
        
    else:
        # PyDeck 3D Hexagon Mapping
        st.write("3D Hexagon density map. Elevation correlates with crime incidence frequency.")
        
        view_state = pdk.ViewState(
            latitude=city_df['latitude'].mean(),
            longitude=city_df['longitude'].mean(),
            zoom=11,
            pitch=55,
            bearing=-15
        )
        
        hexagon_layer = pdk.Layer(
            "HexagonLayer",
            data=city_df,
            get_position="[longitude, latitude]",
            radius=150,
            elevation_scale=50,
            elevation_range=[0, 800],
            extruded=True,
            coverage=0.9,
            pickable=True,
            auto_highlight=True,
            get_fill_color="[200, 30, 255 - elevationValue * 2, 180]"
        )
        
        r = pdk.Deck(
            layers=[hexagon_layer],
            initial_view_state=view_state,
            tooltip={"text": "Crime Density Zone\nIncident Count Density scale applies"},
            map_style="mapbox://styles/mapbox/dark-v9"
        )
        
        st.pydeck_chart(r)

# ==================== TAB 2: HOTSPOT DETECTION ====================
with tab_hotspot:
    st.markdown("### 🎯 Real-Time Threat Hotspot Classifier")
    st.write("Execute live inference on the Random Forest models using spatial and temporal coordinates.")

    col_h_left, col_h_right = st.columns([1, 1])
    
    with col_h_left:
        st.markdown("<div class='glass-panel glow-cyan'>", unsafe_allow_html=True)
        st.subheader("🛡️ Prediction Parameters Form")
        
        # Additional feature sliders for user input
        pred_hour = st.slider("Time of Event (Hour Range)", 0, 23, value=12, help="Hour of occurrence (24hr clock format)")
        pred_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        pred_month = st.slider("Month of Calendar Year", 1, 12, value=6)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Trigger prediction button
        run_prediction = st.button("RUN INTEL INFERENCE", type="primary")

    # Perform calculations when buttons are pressed or inputs change
    # Encodings
    encoded_city = label_encoders['city'].transform([selected_city])[0]
    encoded_area = label_encoders['area'].transform([selected_area])[0]
    encoded_crime_type = label_encoders['crime_type'].transform([selected_crime_type])[0]
    encoded_day = label_encoders['day_of_week'].transform([pred_day])[0]
    
    # Input DF
    input_data = pd.DataFrame([{
        'city_encoded': encoded_city,
        'area_encoded': encoded_area,
        'crime_type_encoded': encoded_crime_type,
        'hour': pred_hour,
        'day_of_week_encoded': encoded_day,
        'month': pred_month
    }])
    
    # Pre-calculate coords
    area_coords = city_df[city_df['area'] == selected_area][['latitude', 'longitude']].mean()
    lat = area_coords['latitude']
    lon = area_coords['longitude']
    
    # Predict Cluster risk zone
    scaled_coords = scaler.transform([[lat, lon]])
    cluster_id = kmeans.predict(scaled_coords)[0]
    assigned_risk_zone = risk_mapping[cluster_id]

    with col_h_right:
        st.markdown("<div class='glass-panel glow-pink'>", unsafe_allow_html=True)
        st.subheader("📊 Tactical Analysis Output")
        
        # Hotspot classification prediction
        hotspot_prob = hotspot_classifier.predict_proba(input_data)[0][1]
        hotspot_pred = hotspot_classifier.predict(input_data)[0]
        
        # Risk Forecasting model prediction
        risk_prob = risk_model.predict_proba(input_data)[0][1]
        
        # Format Results
        hotspot_status_text = "⚠️ HOTSPOT DETECTED (HIGH DENSITY)" if hotspot_pred == 1 else "✅ STABLE ZONE (LOW DENSITY)"
        hotspot_color = "#f43f5e" if hotspot_pred == 1 else "#10b981"
        
        st.markdown(f"**Jurisdiction Selected:** {selected_city} - Sector {selected_area}")
        st.markdown(f"**Estimated Risk Zone:** `{assigned_risk_zone}`")
        st.markdown(f"**Core Classifier Status:** <span style='color:{hotspot_color}; font-weight:bold; font-size:1.15rem;'>{hotspot_status_text}</span>", unsafe_allow_html=True)
        st.markdown(f"**Raw Hotspot Probability:** `{hotspot_prob*100:.1f}%`")
        
        # Plot Gauge Chart for Hotspot Risk Score
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = hotspot_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Hotspot Probability Index (%)", 'font': {'color': "#e5e7eb", 'size': 14}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#4b5563"},
                'bar': {'color': hotspot_color},
                'bgcolor': "rgba(30, 41, 59, 0.4)",
                'borderwidth': 2,
                'bordercolor': "#1e293b",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.3)'},
                    {'range': [40, 70], 'color': 'rgba(249, 115, 22, 0.3)'},
                    {'range': [70, 100], 'color': 'rgba(244, 63, 94, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#e5e7eb", 'family': "Share Tech Mono"},
            margin=dict(l=30, r=30, t=30, b=10),
            height=220
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==================== TAB 3: TEMPORAL FORECAST ====================
with tab_forecast:
    st.markdown("### 📈 Chronological Risk Forecasting Dashboard")
    st.write("Explore risk trends based on temporal variations across hours and days of the week.")

    col_fore_left, col_fore_right = st.columns([1, 2])
    
    with col_fore_left:
        st.markdown("<div class='glass-panel glow-orange'>", unsafe_allow_html=True)
        st.subheader("🗓️ Temporal Setup")
        forecast_day = st.selectbox("Simulate Forecast Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key="f_day")
        forecast_month = st.slider("Simulate Month of Year", 1, 12, value=6, key="f_month")
        st.write("Using Random Forest Forecasting to compute dynamic hourly predictions of Crime Risk Score (based on severe incidents likelihood).")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_fore_right:
        # Calculate risk scores across 24 hours for the selected area, crime type, day, month
        hours = list(range(24))
        hourly_risks = []
        hourly_hotspots = []
        
        # Encodings
        enc_city = label_encoders['city'].transform([selected_city])[0]
        enc_area = label_encoders['area'].transform([selected_area])[0]
        enc_crime = label_encoders['crime_type'].transform([selected_crime_type])[0]
        enc_day = label_encoders['day_of_week'].transform([forecast_day])[0]

        for hr in hours:
            # Build input DF
            hr_df = pd.DataFrame([{
                'city_encoded': enc_city,
                'area_encoded': enc_area,
                'crime_type_encoded': enc_crime,
                'hour': hr,
                'day_of_week_encoded': enc_day,
                'month': forecast_month
            }])
            
            # Predict risk prob and hotspot prob
            r_prob = risk_model.predict_proba(hr_df)[0][1]
            hourly_risks.append(r_prob * 100)
            
        # Draw Line Chart using Plotly
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=hours, 
            y=hourly_risks,
            mode='lines+markers',
            name='Risk Index',
            line=dict(color='#00f2fe', width=3),
            marker=dict(size=6, color='#ffffff', symbol='square-open'),
            fill='tozeroy',
            fillcolor='rgba(0, 242, 254, 0.1)'
        ))
        
        fig_trend.update_layout(
            title=f"Hourly Crime Risk Index Trend: Sector {selected_area} ({forecast_day})",
            xaxis=dict(
                title="Hour of Day (24-Hour Format)",
                tickmode='array',
                tickvals=list(range(0, 24, 2)),
                gridcolor='#1e293b'
            ),
            yaxis=dict(
                title="Crime Risk Score (%)",
                range=[0, 100],
                gridcolor='#1e293b'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#e5e7eb", 'family': "Share Tech Mono"},
            margin=dict(l=40, r=20, t=40, b=40),
            height=350
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# ==================== TAB 4: INTELLIGENCE FEED ====================
with tab_feed:
    st.markdown("### 📰 Command Intelligence Insights & Alert Stream")
    
    col_feed_1, col_feed_2 = st.columns([1, 1])
    
    with col_feed_1:
        st.markdown("<div class='glass-panel glow-cyan'>", unsafe_allow_html=True)
        st.subheader("📋 Core Regional Insights")
        
        # Calculate summary statistics for this city
        crime_counts = city_df['crime_type'].value_counts()
        top_crime = crime_counts.index[0]
        top_crime_val = crime_counts.iloc[0]
        
        area_hotspots = city_df.groupby('area')['hotspot_label'].first()
        riskiest_area = city_df['area'].value_counts().index[0]
        safest_area = city_df['area'].value_counts().index[-1]
        
        hour_counts = city_df['hour'].value_counts()
        riskiest_hour = hour_counts.index[0]
        
        # Display insights list
        st.markdown(f"🛡️ **Sector Jurisdiction Profile: {selected_city}**")
        st.write(f"🔹 **Primary Offense Vector:** `{top_crime}` ({top_crime_val} cases)")
        st.write(f"⚠️ **Highest Density Locality:** Sector `{riskiest_area}`")
        st.write(f"🟢 **Minimal Activity Locality:** Sector `{safest_area}`")
        st.write(f"🕒 **Peak Chrono Vulnerability:** `{riskiest_hour:02d}:00 HRS` (max incidents logged)")
        st.write(f"📊 **Jurisdiction Database Size:** `{len(city_df)}` rows logged")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualizing Crime Types by City
        fig_city_crimes = px.bar(
            x=crime_counts.values,
            y=crime_counts.index,
            orientation='h',
            labels={'x': 'Incident Count', 'y': 'Threat Vector'},
            title=f"Threat Vector Volume Analysis for {selected_city}",
            color=crime_counts.values,
            color_continuous_scale='plasma'
        )
        fig_city_crimes.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#e5e7eb", 'family': "Share Tech Mono"},
            margin=dict(l=40, r=20, t=40, b=40),
            height=250,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_city_crimes, use_container_width=True)

    with col_feed_2:
        st.markdown("<div class='glass-panel glow-pink'>", unsafe_allow_html=True)
        st.subheader("📡 Live Encryption Alert Feed (Mocked Recent logs)")
        
        # Filter recently logged data in this city to show simulated feed
        recent_logs = city_df.sample(min(len(city_df), 5), random_state=123)
        
        for idx, row in recent_logs.iterrows():
            severity_badge = "🔴 HIGH" if row['severity_level'] == "High" else ("🟡 MED" if row['severity_level'] == "Medium" else "🟢 LOW")
            st.markdown(f"""
            📌 **ALERT ID:** `{row['crime_id']}` | **SEV:** {severity_badge}  
            📍 **Sector:** `{row['area']}` | **Type:** `{row['crime_type']}`  
            ⏱️ **Logged:** `{row['date'].strftime('%Y-%m-%d')} {row['time']}` | **Status:** `{row['crime_status']}`  
            ---
            """)
        st.markdown("</div>", unsafe_allow_html=True)

# ==================== TAB 5: AI EXPLANATION & VIVA PREP ====================
with tab_ai:
    st.markdown("### 🧠 AI Explanations & Project Defense Preparation")
    
    st.info("💡 **Student Viva Tip:** Use this tab to align with your project examiner on the core algorithmic and architectural workflows!")

    with st.expander("📚 1. How does the Hotspot Classification Model function?", expanded=True):
        st.markdown("""
        * **Target Variable:** The model predicts `hotspot_label` which is a binary classification ($1$ representing a Hotspot, $0$ representing a Non-Hotspot).
        * **Algorithm:** **Random Forest Classifier**. It operates by constructing a multitude of decision trees during training and outputting the mode of the classes (classification) of the individual trees.
        * **Feature Input Matrix:** `[City, Area, Crime Type, Hour, Day of Week, Month]`.
        * **Calculation details:** Inside the Jupyter notebook, we group crimes by area and city to check incident volume. Areas with incident counts in the **top 30%** (70th percentile threshold) are marked as hotspots. The classifier learns the patterns linking variables like the hour of the day and crime types to identify if an incident location has reached the hotspot threshold.
        """)

    with st.expander("📚 2. How is the Future Crime Risk Score predicted?"):
        st.markdown("""
        * **Target Definition:** We engineered a binary flag `high_risk_flag` which is $1$ if `severity_level` is **High** or **Medium**, and $0$ if it is **Low**.
        * **Risk Score Output:** The Streamlit application invokes `predict_proba()[:, 1]` on the trained model. This extracts the model's confidence probability (ranging from $0.0$ to $1.0$, converted to $0\%$ - $100\%$) that a crime in the given space-temporal window will be severe.
        * **Advantage:** Unlike simple deterministic metrics, this probabilistic forecast models spatial-temporal vulnerability dynamically based on input coordinates and timing.
        """)

    with st.expander("📚 3. What is the role of K-Means Clustering?"):
        st.markdown("""
        * **Mathematical Concept:** K-Means partitions the geographical coordinates (Latitude & Longitude) into $K=3$ clusters by minimizing the squared Euclidean distances between data points and cluster centroids.
        * **Coordinates Scaling:** Prior to clustering, coordinates are standardized using `StandardScaler` to ensure Latitude and Longitude have a mean of $0$ and variance of $1$. Without scaling, coordinate range differences can bias the cluster distances.
        * **Risk Zone Labeling:** Once clustering finishes, we compute the total crime count in each cluster. The cluster containing the maximum crime incidents is classified as **High Risk Zone**, the second as **Medium Risk Zone**, and the last as **Low Risk Zone**.
        """)

    with st.expander("📚 4. Project File Structure Description"):
        st.markdown("""
        * `dataset.csv`: Contains 6,000 synthetically generated realistic crime logs.
        * `notebook.ipynb`: Implements data loading, cleanups, Seaborn/Matplotlib visualizations, feature engineering, K-Means clustering, and Random Forest classifier training.
        * `app.py`: Standard Streamlit script displaying maps and loading saved joblib models to execute live inputs.
        * `*.joblib` files: Serialized files exporting models, scalers, and column headers to avoid re-training when the Streamlit app loads.
        """)

# Footer Command Line log
st.write("---")
st.code("CONSOLE_LOG: SYSTEM STATUS STABLE. SECURITY INTERFACE FULLY ARMED. MONITORING ACTIVE.")
