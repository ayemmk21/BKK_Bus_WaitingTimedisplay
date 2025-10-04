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
    page_title="Bangkok Traffic Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration for large datasets
MAX_ROWS_TO_LOAD = 100000  # Limit number of rows to load
SAMPLE_SIZE_FOR_DISPLAY = 10000  # Sample size for visualizations
CHUNK_SIZE = 50000  # Chunk size for reading large files

# Hugging Face dataset URLs
HUGGINGFACE_BASE = "https://huggingface.co/datasets/Ayemm/BKK_Bus_Data/resolve/main/"

# Main data files
TRAFFIC_DATA_URL = HUGGINGFACE_BASE + "traffic.csv"
CONGESTION_DATA_URL = HUGGINGFACE_BASE + "congestion_zones.csv"

# Alternative/additional data sources
BUS_ROUTES_URL = HUGGINGFACE_BASE + "bangkok_bus_routes.csv"
BUS_STOPS_URL = HUGGINGFACE_BASE + "cleaned_bus_stops_file.csv"
ROUTE_SUMMARY_URL = HUGGINGFACE_BASE + "predicted_route_times_summary.csv"

# Data loading functions with optimization for large files
@st.cache_data(show_spinner="Loading traffic data...", ttl=3600)
def load_traffic_from_url(url: str, max_rows: int = MAX_ROWS_TO_LOAD) -> pd.DataFrame | None:
    """Load traffic data from Hugging Face with row limit"""
    if not url:
        return None
    try:
        st.info(f"📥 Loading traffic data from Hugging Face...")
        # Load only specified number of rows for large datasets
        df = pd.read_csv(url, parse_dates=["timestamp"], nrows=max_rows)
        
        # Verify required columns exist
        required_cols = ["lat", "lon", "speed", "timestamp"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        st.success(f"✅ Loaded {len(df):,} rows from traffic.csv")
        return df
    except Exception as e:
        st.error(f"Error loading traffic data: {e}")
        return None

@st.cache_data(show_spinner="Loading congestion data...", ttl=3600)
def load_congestion_from_url(url: str) -> pd.DataFrame | None:
    """Load congestion data from Hugging Face"""
    if not url:
        return None
    try:
        st.info(f"📥 Loading congestion zones from Hugging Face...")
        df = pd.read_csv(url)
        
        # Verify required columns exist
        required_cols = ["center_lat", "center_lon", "severity", "avg_speed"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        st.success(f"✅ Loaded {len(df):,} congestion zones")
        return df
    except Exception as e:
        st.error(f"Error loading congestion data: {e}")
        return None

@st.cache_data(show_spinner="Loading traffic data from local file...", ttl=3600)
def load_traffic_local(max_rows: int = MAX_ROWS_TO_LOAD) -> pd.DataFrame | None:
    """Load traffic data from local file with chunking for large files"""
    p = "data/traffic.csv"
    if os.path.exists(p):
        try:
            # Get file size to determine loading strategy
            file_size_mb = os.path.getsize(p) / (1024 * 1024)
            
            if file_size_mb > 100:  # If file is larger than 100MB
                st.info(f"📊 Large file detected ({file_size_mb:.1f} MB). Loading optimized sample...")
                # Load in chunks and sample
                chunks = []
                for chunk in pd.read_csv(p, parse_dates=["timestamp"], chunksize=CHUNK_SIZE):
                    # Sample from each chunk to maintain data distribution
                    sample_size = min(len(chunk), max_rows // 10)
                    chunks.append(chunk.sample(n=sample_size, random_state=42))
                    if sum(len(c) for c in chunks) >= max_rows:
                        break
                df = pd.concat(chunks, ignore_index=True)
            else:
                # Load normally with row limit
                df = pd.read_csv(p, parse_dates=["timestamp"], nrows=max_rows)
            
            return df
        except Exception as e:
            st.error(f"Error loading local traffic data: {e}")
            return None
    return None

@st.cache_data(show_spinner="Loading congestion data from local file...", ttl=3600)
def load_congestion_local() -> pd.DataFrame | None:
    """Load congestion data from local file"""
    p = "data/congestion.csv"
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            return df
        except Exception as e:
            st.error(f"Error loading local congestion data: {e}")
            return None
    return None

def load_data():
    """Load data from URL (priority) or local file (fallback)"""
    # Try loading traffic data
    traffic_df = load_traffic_from_url(TRAFFIC_DATA_URL)
    if traffic_df is None:
        traffic_df = load_traffic_local()
    
    # Try loading congestion data
    congestion_df = load_congestion_from_url(CONGESTION_DATA_URL)
    if congestion_df is None:
        congestion_df = load_congestion_local()
    
    return traffic_df, congestion_df

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


# Auto-load data
with st.spinner("Loading data..."):
    traffic_df, congestion_df = load_data()

# Show data info if loaded
if traffic_df is not None:
    st.sidebar.success(f"✅ Traffic data loaded: {len(traffic_df):,} rows")
if congestion_df is not None:
    st.sidebar.success(f"✅ Congestion data loaded: {len(congestion_df):,} rows")


# Check if data is loaded
if traffic_df is None or congestion_df is None:
    st.warning("No data could be loaded from Hugging Face or local files.")
    st.info("""
    **Configured Data Sources:**
    - **Traffic Data:** `traffic.csv` from Hugging Face
    - **Congestion Zones:** `congestion_zones.csv` from Hugging Face
    
    **Available files in your dataset:**
    - bangkok_bus_routes.csv / .xlsx
    - bus_stop_locations.csv
    - cleaned_bus_routes_file.csv / .xlsx
    - cleaned_bus_stops_file.csv / .xlsx
    - congestion_zones.csv / .xlsx
    - traffic.csv
    - predicted_route_times_summary.csv
    
    **Fallback Option:**
    - Create a `data/` folder in your project directory
    - Add `traffic.csv` and `congestion_zones.csv` files locally
    
    **Manual Upload:**
    - Use the sidebar file uploaders
    
    **Note:** For large datasets, the app automatically:
    - Loads only the first {MAX_ROWS_TO_LOAD:,} rows
    - Caches data for 1 hour
    - Shows loading progress
    """)
    
    st.subheader("Data Source Status")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Traffic URL:\n{TRAFFIC_DATA_URL}")
    with col2:
        st.info(f"Congestion URL:\n{CONGESTION_DATA_URL}")
    
    st.stop()


# Speed threshold
speed_threshold = st.sidebar.slider(
    "Speed Threshold (km/h)",
    min_value=0,
    max_value=120,
    value=30,
    step=5
)

# Main content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", 
    "🗺️ Geographic Analysis", 
    "⏰ Temporal Patterns", 
    "🚦 Congestion Analysis",
    "📊 Model Insights",
    "🛣️ Route Analysis"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.header("Overview Statistics")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{len(traffic_df):,}")
    
    with col2:
        avg_speed = traffic_df['speed'].mean()
        st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
    
    with col3:
        if 'near_congestion' in traffic_df.columns:
            congestion_pct = (traffic_df['near_congestion'].sum() / len(traffic_df)) * 100
            st.metric("Near Congestion", f"{congestion_pct:.1f}%")
        else:
            st.metric("Near Congestion", "N/A")
    
    with col4:
        slow_traffic = (traffic_df['speed'] < speed_threshold).sum()
        slow_pct = (slow_traffic / len(traffic_df)) * 100
        st.metric(f"Speed < {speed_threshold} km/h", f"{slow_pct:.1f}%")
    
    with col5:
        unique_vehicles = traffic_df['VehicleID'].nunique() if 'VehicleID' in traffic_df.columns else 'N/A'
        st.metric("Unique Vehicles", f"{unique_vehicles:,}" if isinstance(unique_vehicles, int) else unique_vehicles)
    
    st.markdown("---")
    
    # Speed distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Speed Distribution")
        fig = px.histogram(
            traffic_df, 
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
        if 'hour' in traffic_df.columns:
            hourly_stats = traffic_df.groupby('hour')['speed'].agg(['mean', 'median', 'std']).reset_index()
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
# TAB 3: TEMPORAL PATTERNS
# ============================================================================
with tab3:
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
    if 'near_congestion' in traffic_df.columns:
        st.subheader("Speed Comparison: Near vs Far from Congestion Zones")
        
        near = traffic_df[traffic_df['near_congestion'] == 1]['speed']
        far = traffic_df[traffic_df['near_congestion'] == 0]['speed']
        
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

# ============================================================================
# TAB 6: ROUTE ANALYSIS
# ============================================================================
with tab6:
    st.header("Route-Level Analysis")
    
    # File uploader for route data
    st.subheader("📁 Upload Route Data")
    route_file = st.file_uploader("Upload Route Data CSV", type=['csv'], key='route_uploader')
    
    if route_file is not None:
        route_df = pd.read_csv(route_file)
        
        st.subheader("Route-by-Route Details")
        
        required_cols = ['ref', 'name', 'num_stops', 'total_distance_km', 
                         'avg_predicted_speed_kmh', 'total_predicted_travel_time_min']
        missing_cols = [col for col in required_cols if col not in route_df.columns]
        
        if missing_cols:
            st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            st.info("""
            **Expected columns:**
            - `ref`: Route reference number or ID  
            - `name`: Route name  
            - `num_stops`: Number of stops  
            - `total_distance_km`: Total route distance (km)  
            - `avg_predicted_speed_kmh`: Average predicted speed (km/h)  
            - `total_predicted_travel_time_min`: Total travel time (minutes)
            """)
        else:
            # Format dataframe for display
            display_df = route_df.rename(columns={
                'ref': 'Route Ref',
                'name': 'Route Name',
                'num_stops': 'Stops',
                'total_distance_km': 'Distance (km)',
                'avg_predicted_speed_kmh': 'Avg Speed (km/h)',
                'total_predicted_travel_time_min': 'Travel Time (min)'
            })
            
            def style_route_table(df):
                styled = df.style.format({
                    'Distance (km)': '{:.2f}',
                    'Avg Speed (km/h)': '{:.2f}',
                    'Travel Time (min)': '{:.2f}'
                })
                
                styled = styled.background_gradient(
                    subset=['Avg Speed (km/h)'],
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=df['Avg Speed (km/h)'].max()
                )
                
                styled = styled.background_gradient(
                    subset=['Travel Time (min)'],
                    cmap='RdYlGn_r',
                    vmin=0,
                    vmax=df['Travel Time (min)'].max()
                )
                return styled
            
            st.dataframe(style_route_table(display_df), use_container_width=True, height=400)
            
            # Summary
            st.markdown("---")
            st.subheader("Overall Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            total_distance = route_df['total_distance_km'].sum()
            total_time = route_df['total_predicted_travel_time_min'].sum()
            avg_speed = route_df['avg_predicted_speed_kmh'].mean()
            total_routes = len(route_df)
            
            with col1:
                st.metric("Total Distance", f"{total_distance:.2f} km")
            with col2:
                st.metric("Total Travel Time", f"{total_time:.2f} min")
            with col3:
                st.metric("Average Speed", f"{avg_speed:.2f} km/h")
            with col4:
                st.metric("Number of Routes", f"{total_routes}")
            
            # Visualizations
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Average Speed by Route")
                fig = px.bar(
                    route_df,
                    x='ref',
                    y='avg_predicted_speed_kmh',
                    title="Average Predicted Speed per Route",
                    labels={'ref': 'Route', 'avg_predicted_speed_kmh': 'Speed (km/h)'},
                    color='avg_predicted_speed_kmh',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Total Travel Time by Route")
                fig = px.bar(
                    route_df,
                    x='ref',
                    y='total_predicted_travel_time_min',
                    title="Total Travel Time per Route",
                    labels={'ref': 'Route', 'total_predicted_travel_time_min': 'Time (min)'},
                    color='total_predicted_travel_time_min',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance analysis
            st.markdown("---")
            st.subheader("Performance Analysis")
            
            slowest = route_df.loc[route_df['avg_predicted_speed_kmh'].idxmin()]
            longest = route_df.loc[route_df['total_predicted_travel_time_min'].idxmax()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.warning(f"""
                **⚠️ Slowest Route**
                - Route: {slowest['ref']} ({slowest['name']})
                - Speed: {slowest['avg_predicted_speed_kmh']:.2f} km/h
                - Distance: {slowest['total_distance_km']:.2f} km
                """)
            
            with col2:
                st.error(f"""
                **🚨 Longest Travel Time**
                - Route: {longest['ref']} ({longest['name']})
                - Time: {longest['total_predicted_travel_time_min']:.2f} min
                - Distance: {longest['total_distance_km']:.2f} km
                """)
    else:
        st.info("""
        📤 **Upload a route-level CSV file to view analysis**
        
        **Expected CSV format:**
        ```
        ref,name,num_stops,total_distance_km,avg_predicted_speed_kmh,total_predicted_travel_time_min
        1,Route A,12,22.50,35.60,38.00
        2,Route B,8,18.90,42.00,27.00
        ...
        ```
        """)


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Bangkok Traffic Analysis Dashboard | Built with Streamlit & Plotly</p>
    </div>
""", unsafe_allow_html=True)