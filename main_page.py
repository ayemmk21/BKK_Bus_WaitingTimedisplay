import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import os

# Page configuration
st.set_page_config(
    page_title="Traffic Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Bangkok Traffic Analysis Dashboard")
st.sidebar.success("Select a page above.")

# Hugging Face dataset URLs
HUGGINGFACE_BASE = "https://huggingface.co/datasets/Ayemm/BKK_Bus_Data/resolve/main/"

# Main data files
TRAFFIC_DATA_URL = HUGGINGFACE_BASE + "traffic.csv"
CONGESTION_DATA_URL = HUGGINGFACE_BASE + "congestion_zones.csv"

# Additional data files
BUS_ROUTES_URL = HUGGINGFACE_BASE + "cleaned_bus_routes_file.csv"
BUS_STOPS_URL = HUGGINGFACE_BASE + "cleaned_bus_stops_file.csv"
ROUTE_SUMMARY_URL = HUGGINGFACE_BASE + "predicted_route_times_summary.csv"


# --- Remote data loader with caching ---

@st.cache_data(show_spinner=False, ttl=3600)
def load_csv_from_url(url: str, required_cols: list[str] | None = None, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    """Load a full CSV file from Hugging Face and validate required columns."""
    if not url:
        return None
    try:
        df = pd.read_csv(url, parse_dates=parse_dates)
        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing} in {os.path.basename(url)}")
                return None
        st.success(f"✅ Loaded {len(df):,} rows from {os.path.basename(url)}")
        return df
    except Exception as e:
        st.error(f"❌ Error loading {os.path.basename(url)}: {e}")
        return None


# --- Local fallback loaders with caching ---

@st.cache_data(show_spinner=False, ttl=3600)
def load_traffic_local() -> pd.DataFrame | None:
    """Load traffic data from local file as fallback."""
    p = "data/traffic.csv"
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, parse_dates=["timestamp"])
            st.success(f"✅ Loaded {len(df):,} rows from local traffic.csv")
            return df
        except Exception as e:
            st.error(f"❌ Error loading local traffic data: {e}")
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_congestion_local() -> pd.DataFrame | None:
    """Load congestion data from local file as fallback."""
    p = "data/congestion.csv"
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            st.success(f"✅ Loaded {len(df):,} rows from local congestion.csv")
            return df
        except Exception as e:
            st.error(f"❌ Error loading local congestion data: {e}")
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_bus_routes_local() -> pd.DataFrame | None:
    """Load bus routes data from local file as fallback."""
    p = "data/bangkok_bus_routes.csv"
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            st.success(f"✅ Loaded {len(df):,} rows from local bangkok_bus_routes.csv")
            return df
        except Exception as e:
            st.error(f"❌ Error loading local bus routes data: {e}")
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_bus_stops_local() -> pd.DataFrame | None:
    """Load bus stops data from local file as fallback."""
    p = "data/cleaned_bus_stops_file.csv"
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            st.success(f"✅ Loaded {len(df):,} rows from local cleaned_bus_stops_file.csv")
            return df
        except Exception as e:
            st.error(f"❌ Error loading local bus stops data: {e}")
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_route_summary_local() -> pd.DataFrame | None:
    """Load route summary data from local file as fallback."""
    p = "data/predicted_route_times_summary.csv"
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            st.success(f"✅ Loaded {len(df):,} rows from local predicted_route_times_summary.csv")
            return df
        except Exception as e:
            st.error(f"❌ Error loading local route summary data: {e}")
    return None


# --- Main data loader function ---

def load_data():
    """
    Load all required data from Hugging Face, with local file fallback.
    
    Returns:
        tuple: (traffic_df, congestion_df, bus_routes_df, bus_stops_df, route_summary_df)
    """
    # Load traffic data (required)
    traffic_df = load_csv_from_url(
        TRAFFIC_DATA_URL,
        required_cols=["lat", "lon", "speed", "timestamp"],
        parse_dates=["timestamp"]
    )
    if traffic_df is None:
        st.warning("⚠️ Failed to load from Hugging Face, trying local fallback...")
        traffic_df = load_traffic_local()
    
    # Load congestion data (required)
    congestion_df = load_csv_from_url(
        CONGESTION_DATA_URL,
        required_cols=["center_lat", "center_lon", "severity", "avg_speed"]
    )
    if congestion_df is None:
        st.warning("⚠️ Failed to load from Hugging Face, trying local fallback...")
        congestion_df = load_congestion_local()
    
    # Load bus routes data (optional)
    bus_routes_df = load_csv_from_url(BUS_ROUTES_URL)
    if bus_routes_df is None:
        bus_routes_df = load_bus_routes_local()
    
    # Load bus stops data (optional)
    bus_stops_df = load_csv_from_url(BUS_STOPS_URL)
    if bus_stops_df is None:
        bus_stops_df = load_bus_stops_local()
    
    # Load route summary data (optional)
    route_summary_df = load_csv_from_url(ROUTE_SUMMARY_URL)
    if route_summary_df is None:
        route_summary_df = load_route_summary_local()
    
    return traffic_df, congestion_df, bus_routes_df, bus_stops_df, route_summary_df


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# Auto-load data
with st.spinner("Loading data..."):
    traffic_df, congestion_df, bus_routes_df, bus_stops_df, route_summary_df = load_data()


# Check if data is loaded
if traffic_df is None or congestion_df is None:
    st.warning("No data could be loaded from Hugging Face or local files.")
    st.info("""
    **Configured Data Sources:**
    - **Traffic Data:** `traffic.csv` from Hugging Face
    - **Congestion Zones:** `congestion_zones.csv` from Hugging Face
    
    **Available files in your dataset:**
    - cleaned_bus_routes_file.csv 
    - cleaned_bus_stops_file.csv
    - congestion_zones.csv 
    - traffic.csv
    - predicted_route_times_summary.csv
    
    **Fallback Option:**
    - Create a `data/` folder in your project directory
    - Add `traffic.csv` and `congestion_zones.csv` files locally
    
    """)
    
    st.subheader("Data Source Status")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Traffic URL:\n{TRAFFIC_DATA_URL}")
    with col2:
        st.info(f"Congestion URL:\n{CONGESTION_DATA_URL}")
    
    st.stop()


# Main content
tab1, tab2, tab3 = st.tabs([
    "🗺️ Geographic Analysis", 
    "⏰ Temporal Patterns", 
    "📊 Model Insights",
])

# ============================================================================
# TAB 1: GEOGRAPHIC ANALYSIS
# ============================================================================
with tab1:
    st.header("Geographic Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Traffic Speed Heatmap")
        
        # Sample data for performance
        sample_size = min(10000, len(traffic_df))
        df_sample = traffic_df.sample(n=sample_size, random_state=42)
        
        # Create map centered on Bangkok
        bangkok_center = [13.7563, 100.5018]
        m = folium.Map(location=bangkok_center, zoom_start=11, tiles='OpenStreetMap')
        
        # Add congestion zones
        for _, zone in congestion_df.iterrows():
            color = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}.get(zone['severity'], 'gray')
            folium.Circle(
                location=[zone['center_lat'], zone['center_lon']],
                radius=500,
                popup=f"Zone {zone['zone_id']}<br>Severity: {zone['severity']}<br>Avg Speed: {zone['avg_speed']:.1f} km/h",
                color=color,
                fill=True,
                fillOpacity=0.3
            ).add_to(m)
        
        # Add traffic points (color by speed)
        for _, row in df_sample.iterrows():
            speed = row['speed']
            if speed < 20:
                color = 'red'
            elif speed < 40:
                color = 'orange'
            elif speed < 60:
                color = 'yellow'
            else:
                color = 'green'
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=2,
                color=color,
                fill=True,
                fillOpacity=0.6,
                popup=f"Speed: {speed:.1f} km/h"
            ).add_to(m)
        
        folium_static(m, width=800, height=600)
    
    with col2:
        st.subheader("Congestion Zones")
        st.dataframe(
            congestion_df[['zone_id', 'severity', 'avg_speed', 'size']].sort_values('severity'),
            height=300
        )
        
        st.subheader("Speed by Distance from Center")
        if 'distance_from_center' in traffic_df.columns:
            fig = px.scatter(
                traffic_df.sample(n=min(5000, len(traffic_df))),
                x='distance_from_center',
                y='speed',
                color='speed',
                title="Speed vs Distance from Center",
                labels={'distance_from_center': 'Distance from Center', 'speed': 'Speed (km/h)'},
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: TEMPORAL PATTERNS
# ============================================================================
with tab2:
    st.header("Temporal Traffic Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Traffic by Hour of Day")
        if 'hour' in traffic_df.columns:
            hourly = traffic_df.groupby('hour').agg({
                'speed': ['mean', 'count']
            }).reset_index()
            hourly.columns = ['hour', 'avg_speed', 'count']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=hourly['hour'], y=hourly['count'], name='Traffic Volume', marker_color='lightblue'),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=hourly['hour'], y=hourly['avg_speed'], name='Avg Speed', 
                          mode='lines+markers', line=dict(color='red', width=3)),
                secondary_y=True
            )
            fig.update_xaxes(title_text="Hour of Day")
            fig.update_yaxes(title_text="Traffic Volume", secondary_y=False)
            fig.update_yaxes(title_text="Average Speed (km/h)", secondary_y=True)
            fig.update_layout(title="Traffic Volume and Speed by Hour")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Traffic by Day of Week")
        if 'day_of_week' in traffic_df.columns:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily = traffic_df.groupby('day_of_week')['speed'].mean().reset_index()
            daily['day_name'] = daily['day_of_week'].apply(lambda x: days[x])
            
            fig = px.bar(
                daily,
                x='day_name',
                y='speed',
                title="Average Speed by Day of Week",
                labels={'day_name': 'Day', 'speed': 'Avg Speed (km/h)'},
                color='speed',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Rush hour analysis
    if 'is_rush_hour' in traffic_df.columns:
        st.subheader("Rush Hour vs Non-Rush Hour Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        rush_hour_data = traffic_df[traffic_df['is_rush_hour'] == 1]
        non_rush_data = traffic_df[traffic_df['is_rush_hour'] == 0]
        
        with col1:
            st.metric(
                "Rush Hour Avg Speed",
                f"{rush_hour_data['speed'].mean():.1f} km/h",
                delta=f"{rush_hour_data['speed'].mean() - traffic_df['speed'].mean():.1f} km/h"
            )
        
        with col2:
            st.metric(
                "Non-Rush Hour Avg Speed",
                f"{non_rush_data['speed'].mean():.1f} km/h",
                delta=f"{non_rush_data['speed'].mean() - traffic_df['speed'].mean():.1f} km/h"
            )
        
        with col3:
            st.metric(
                "Speed Difference",
                f"{abs(rush_hour_data['speed'].mean() - non_rush_data['speed'].mean()):.1f} km/h"
            )

# ============================================================================
# TAB 3: MODEL INSIGHTS
# ============================================================================
with tab3:
    st.header("Model Insights & Features")
    
    # Feature correlation
    st.subheader("Feature Correlations with Speed")
    
    feature_cols = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 
                   'distance_from_center', 'near_congestion', 'distance_to_congestion']
    available_features = [col for col in feature_cols if col in traffic_df.columns]
    
    if available_features:
        correlations = traffic_df[available_features + ['speed']].corr()['speed'].drop('speed').sort_values()
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title="Feature Correlation with Speed",
            labels={'x': 'Correlation', 'y': 'Feature'},
            color=correlations.values,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance Preview")
        st.info("""
        **Key Features for Traffic Prediction:**
        - 🕐 Temporal: hour, day_of_week, is_rush_hour
        - 📍 Spatial: lat, lon, distance_from_center
        - 🚦 Congestion: near_congestion, congestion_severity
        - 📊 Historical: location_avg_speed, hour_avg_speed
        """)
    
    with col2:
        st.subheader("Data Quality Metrics")
        st.metric("Missing Values", f"{traffic_df.isnull().sum().sum()}")
        st.metric("Duplicate Records", f"{traffic_df.duplicated().sum()}")
        if 'speed' in traffic_df.columns:
            st.metric("Speed Outliers (>120 km/h)", f"{(traffic_df['speed'] > 120).sum()}")


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Bangkok Traffic Analysis Dashboard | Built with Streamlit & Plotly</p>
    </div>
""", unsafe_allow_html=True)