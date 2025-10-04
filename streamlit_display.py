import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static

# Page configuration
st.set_page_config(
    page_title="Bangkok Traffic Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Title
st.markdown('<p class="main-header">🚗 Bangkok Traffic Analysis Dashboard</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Dashboard Controls")
st.sidebar.markdown("---")

# File uploaders
st.sidebar.subheader("📁 Upload Data Files")
traffic_file = st.sidebar.file_uploader("Upload Traffic Data (CSV)", type=['csv'])
congestion_file = st.sidebar.file_uploader("Upload Congestion Zones (CSV)", type=['csv'])

# Load data
@st.cache_data
def load_data(traffic_file, congestion_file):
    if traffic_file is not None:
        traffic_df = pd.read_csv(traffic_file)
        traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
    else:
        traffic_df = None
    
    if congestion_file is not None:
        congestion_df = pd.read_csv(congestion_file)
    else:
        congestion_df = None
    
    return traffic_df, congestion_df

traffic_df, congestion_df = load_data(traffic_file, congestion_file)

# Check if data is loaded
if traffic_df is None or congestion_df is None:
    st.warning("⚠️ Please upload both Traffic Data and Congestion Zones CSV files to view the dashboard.")
    st.info("""
    **Expected file formats:**
    - **Traffic Data**: Should contain columns like `lat`, `lon`, `speed`, `timestamp`, `hour`, `is_rush_hour`, etc.
    - **Congestion Zones**: Should contain `center_lat`, `center_lon`, `severity`, `avg_speed`, `size`
    """)
    st.stop()

# Filters in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

# Date range filter
if 'timestamp' in traffic_df.columns:
    min_date = traffic_df['timestamp'].min().date()
    max_date = traffic_df['timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter by date
    if len(date_range) == 2:
        mask = (traffic_df['timestamp'].dt.date >= date_range[0]) & (traffic_df['timestamp'].dt.date <= date_range[1])
        traffic_df_filtered = traffic_df[mask]
    else:
        traffic_df_filtered = traffic_df
else:
    traffic_df_filtered = traffic_df

# Hour filter
if 'hour' in traffic_df_filtered.columns:
    hours = sorted(traffic_df_filtered['hour'].unique())
    selected_hours = st.sidebar.multiselect(
        "Select Hours",
        options=hours,
        default=hours
    )
    traffic_df_filtered = traffic_df_filtered[traffic_df_filtered['hour'].isin(selected_hours)]

# Speed threshold
speed_threshold = st.sidebar.slider(
    "Speed Threshold (km/h)",
    min_value=0,
    max_value=120,
    value=30,
    step=5
)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", 
    "🗺️ Geographic Analysis", 
    "⏰ Temporal Patterns", 
    "🚦 Congestion Analysis",
    "📊 Model Insights"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.header("Overview Statistics")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{len(traffic_df_filtered):,}")
    
    with col2:
        avg_speed = traffic_df_filtered['speed'].mean()
        st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
    
    with col3:
        if 'near_congestion' in traffic_df_filtered.columns:
            congestion_pct = (traffic_df_filtered['near_congestion'].sum() / len(traffic_df_filtered)) * 100
            st.metric("Near Congestion", f"{congestion_pct:.1f}%")
        else:
            st.metric("Near Congestion", "N/A")
    
    with col4:
        slow_traffic = (traffic_df_filtered['speed'] < speed_threshold).sum()
        slow_pct = (slow_traffic / len(traffic_df_filtered)) * 100
        st.metric(f"Speed < {speed_threshold} km/h", f"{slow_pct:.1f}%")
    
    with col5:
        unique_vehicles = traffic_df_filtered['VehicleID'].nunique() if 'VehicleID' in traffic_df_filtered.columns else 'N/A'
        st.metric("Unique Vehicles", f"{unique_vehicles:,}" if isinstance(unique_vehicles, int) else unique_vehicles)
    
    st.markdown("---")
    
    # Speed distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Speed Distribution")
        fig = px.histogram(
            traffic_df_filtered, 
            x='speed', 
            nbins=50,
            title="Traffic Speed Distribution",
            labels={'speed': 'Speed (km/h)', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=speed_threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Threshold: {speed_threshold} km/h")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Speed Statistics by Hour")
        if 'hour' in traffic_df_filtered.columns:
            hourly_stats = traffic_df_filtered.groupby('hour')['speed'].agg(['mean', 'median', 'std']).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats['mean'], 
                                    mode='lines+markers', name='Mean', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats['median'], 
                                    mode='lines+markers', name='Median', line=dict(color='green')))
            fig.update_layout(
                title="Average Speed by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Speed (km/h)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: GEOGRAPHIC ANALYSIS
# ============================================================================
with tab2:
    st.header("Geographic Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Traffic Speed Heatmap")
        
        # Sample data for performance
        sample_size = min(10000, len(traffic_df_filtered))
        df_sample = traffic_df_filtered.sample(n=sample_size, random_state=42)
        
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
        if 'distance_from_center' in traffic_df_filtered.columns:
            fig = px.scatter(
                traffic_df_filtered.sample(n=min(5000, len(traffic_df_filtered))),
                x='distance_from_center',
                y='speed',
                color='speed',
                title="Speed vs Distance from Center",
                labels={'distance_from_center': 'Distance from Center', 'speed': 'Speed (km/h)'},
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: TEMPORAL PATTERNS
# ============================================================================
with tab3:
    st.header("Temporal Traffic Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Traffic by Hour of Day")
        if 'hour' in traffic_df_filtered.columns:
            hourly = traffic_df_filtered.groupby('hour').agg({
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
        if 'day_of_week' in traffic_df_filtered.columns:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily = traffic_df_filtered.groupby('day_of_week')['speed'].mean().reset_index()
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
    if 'is_rush_hour' in traffic_df_filtered.columns:
        st.subheader("Rush Hour vs Non-Rush Hour Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        rush_hour_data = traffic_df_filtered[traffic_df_filtered['is_rush_hour'] == 1]
        non_rush_data = traffic_df_filtered[traffic_df_filtered['is_rush_hour'] == 0]
        
        with col1:
            st.metric(
                "Rush Hour Avg Speed",
                f"{rush_hour_data['speed'].mean():.1f} km/h",
                delta=f"{rush_hour_data['speed'].mean() - traffic_df_filtered['speed'].mean():.1f} km/h"
            )
        
        with col2:
            st.metric(
                "Non-Rush Hour Avg Speed",
                f"{non_rush_data['speed'].mean():.1f} km/h",
                delta=f"{non_rush_data['speed'].mean() - traffic_df_filtered['speed'].mean():.1f} km/h"
            )
        
        with col3:
            st.metric(
                "Speed Difference",
                f"{abs(rush_hour_data['speed'].mean() - non_rush_data['speed'].mean()):.1f} km/h"
            )

# ============================================================================
# TAB 4: CONGESTION ANALYSIS
# ============================================================================
with tab4:
    st.header("Congestion Zone Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Congestion Severity Distribution")
        severity_counts = congestion_df['severity'].value_counts()
        fig = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="Distribution of Congestion Severity",
            color=severity_counts.index,
            color_discrete_map={'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Congestion Zone Sizes")
        fig = px.bar(
            congestion_df.sort_values('size', ascending=False).head(10),
            x='zone_id',
            y='size',
            color='severity',
            title="Top 10 Largest Congestion Zones",
            labels={'zone_id': 'Zone ID', 'size': 'Zone Size'},
            color_discrete_map={'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Speed comparison near congestion
    if 'near_congestion' in traffic_df_filtered.columns:
        st.subheader("Speed Comparison: Near vs Far from Congestion Zones")
        
        near = traffic_df_filtered[traffic_df_filtered['near_congestion'] == 1]['speed']
        far = traffic_df_filtered[traffic_df_filtered['near_congestion'] == 0]['speed']
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=near, name='Near Congestion', marker_color='red'))
        fig.add_trace(go.Box(y=far, name='Far from Congestion', marker_color='green'))
        fig.update_layout(
            title="Speed Distribution by Proximity to Congestion Zones",
            yaxis_title="Speed (km/h)"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 5: MODEL INSIGHTS
# ============================================================================
with tab5:
    st.header("Model Insights & Features")
    
    # Feature correlation
    st.subheader("Feature Correlations with Speed")
    
    feature_cols = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 
                   'distance_from_center', 'near_congestion', 'distance_to_congestion']
    available_features = [col for col in feature_cols if col in traffic_df_filtered.columns]
    
    if available_features:
        correlations = traffic_df_filtered[available_features + ['speed']].corr()['speed'].drop('speed').sort_values()
        
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
        st.metric("Missing Values", f"{traffic_df_filtered.isnull().sum().sum()}")
        st.metric("Duplicate Records", f"{traffic_df_filtered.duplicated().sum()}")
        if 'speed' in traffic_df_filtered.columns:
            st.metric("Speed Outliers (>120 km/h)", f"{(traffic_df_filtered['speed'] > 120).sum()}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Bangkok Traffic Analysis Dashboard | Built with Streamlit & Plotly</p>
    </div>
""", unsafe_allow_html=True)