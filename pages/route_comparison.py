import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import folium_static
import folium
from math import radians, sin, cos, sqrt, atan2, asin
import ast
from datetime import datetime, time, timedelta

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Bus Route Optimizer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .section-header {
        background-color: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
    }
    .stDataFrame {
        border: 2px solid #667eea;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== HELPER FUNCTIONS (FROM COLAB) ==========
def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points using Haversine formula"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def parse_coordinates(coord_str):
    """Parse coordinate string to list of [lon, lat] pairs"""
    if isinstance(coord_str, str):
        try:
            # Remove brackets and split
            coords_str = coord_str.strip('[]')
            coords = []
            for pair_str in coords_str.split('], ['):
                lon_str, lat_str = pair_str.strip('[]').split(',')
                coords.append([float(lon_str), float(lat_str)])
            return coords
        except Exception as e:
            st.warning(f"Error parsing coordinates: {e}")
            return []
    elif isinstance(coord_str, list):
        return coord_str
    else:
        return []

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate perpendicular distance from point to line segment"""
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return haversine(px, py, x1, y1)
    
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return haversine(px, py, closest_x, closest_y)

def is_point_near_route(point_lat, point_lon, route_coords, threshold_km=0.5):
    """Check if a point is near a route"""
    if not route_coords or len(route_coords) < 2:
        return False, float('inf')
    
    min_distance = float('inf')
    
    for i in range(len(route_coords) - 1):
        lon1, lat1 = route_coords[i]
        lon2, lat2 = route_coords[i + 1]
        
        dist = point_to_line_distance(point_lon, point_lat, lon1, lat1, lon2, lat2)
        min_distance = min(min_distance, dist)
        
        if min_distance <= threshold_km:
            return True, min_distance
    
    return min_distance <= threshold_km, min_distance

def find_nearest_segment(lat, lon, route_coords):
    """Find which segment index is closest to the given point"""
    min_distance = float('inf')
    nearest_segment = 0
    
    for i in range(len(route_coords) - 1):
        lon1, lat1 = route_coords[i]
        lon2, lat2 = route_coords[i + 1]
        
        dist = point_to_line_distance(lon, lat, lon1, lat1, lon2, lat2)
        
        if dist < min_distance:
            min_distance = dist
            nearest_segment = i
    
    return nearest_segment

def find_connecting_routes(origin_lat, origin_lon, dest_lat, dest_lon,
                           routes_df, max_distance_km=0.5):
    """Find routes that connect origin and destination"""
    connecting_routes = []
    
    for idx, route in routes_df.iterrows():
        route_id = str(route['route_id'])
        coords = parse_coordinates(route['coordinates'])
        
        if not coords or len(coords) < 2:
            continue
        
        near_origin, dist_origin = is_point_near_route(origin_lat, origin_lon,
                                                        coords, max_distance_km)
        
        near_dest, dist_dest = is_point_near_route(dest_lat, dest_lon,
                                                     coords, max_distance_km)
        
        if near_origin and near_dest:
            origin_segment = find_nearest_segment(origin_lat, origin_lon, coords)
            dest_segment = find_nearest_segment(dest_lat, dest_lon, coords)
            
            # Only include if route goes from origin to dest (not backwards)
            if origin_segment < dest_segment:
                connecting_routes.append({
                    'route_id': route_id,
                    'distance_from_origin': dist_origin,
                    'distance_from_dest': dist_dest,
                    'route_name': route.get('name', f'Route {route_id}'),
                    'total_distance_km': route.get('total_distance_km', 0),
                    'coordinates': coords
                })
    
    return connecting_routes

def make_features_for_time(lat, lon, timestamp, heading=0):
    """Create features for time prediction"""
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0
    
    # Bangkok center
    bangkok_center_lat, bangkok_center_lon = 13.7563, 100.5018
    distance_from_center = sqrt(
        (lat - bangkok_center_lat)**2 +
        (lon - bangkok_center_lon)**2
    )
    
    # Grid cells (assuming 0.01 degree grid)
    lat_grid = round(lat * 100) / 100
    lon_grid = round(lon * 100) / 100
    
    return {
        'lat': lat,
        'lon': lon,
        'heading': heading,
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour,
        'lat_grid': lat_grid,
        'lon_grid': lon_grid,
        'distance_from_center': distance_from_center,
    }

def predict_route_metrics(route_info, departure_time):
    """
    Predict route metrics based on coordinates and time
    Using simplified calculation matching Colab logic
    """
    coords = route_info['coordinates']
    
    # Calculate segment distances
    segment_distances = []
    for i in range(len(coords) - 1):
        dist = haversine(coords[i][0], coords[i][1],
                        coords[i+1][0], coords[i+1][1])
        segment_distances.append(dist)
    
    total_distance = sum(segment_distances)
    
    # Predict speeds based on time and location
    hour = departure_time.hour
    is_rush_hour = (7 <= hour <= 9) or (16 <= hour <= 19)
    
    # Simulate speed predictions (replace with actual model when available)
    if is_rush_hour:
        base_speed = np.random.uniform(18, 25)  # Slower during rush hour
    else:
        base_speed = np.random.uniform(22, 30)  # Faster otherwise
    
    # Add variability per segment
    segment_speeds = []
    for i, dist in enumerate(segment_distances):
        # Add some randomness to simulate traffic variation
        speed_variation = np.random.uniform(-3, 3)
        segment_speed = max(base_speed + speed_variation, 10)  # Min 10 km/h
        segment_speeds.append(segment_speed)
    
    # Calculate time for each segment
    segment_times = []
    cumulative_time = 0
    
    for i, (dist, speed) in enumerate(zip(segment_distances, segment_speeds)):
        time_min = (dist / speed) * 60
        cumulative_time += time_min
        segment_times.append({
            'segment': i + 1,
            'distance_km': dist,
            'predicted_speed': speed,
            'time_min': time_min,
            'cumulative_time': cumulative_time
        })
    
    total_time = cumulative_time
    avg_speed = (total_distance / (total_time / 60)) if total_time > 0 else 0
    slowest_speed = min(segment_speeds) if segment_speeds else 0
    
    return {
        'total_distance_km': total_distance,
        'predicted_time_min': total_time,
        'avg_speed_kmh': avg_speed,
        'slowest_segment_speed': slowest_speed,
        'num_segments': len(segment_distances),
        'segment_details': pd.DataFrame(segment_times)
    }

def compare_routes_streamlit(origin_lat, origin_lon, dest_lat, dest_lon,
                             departure_time, routes_df, max_search_distance=0.5, top_n=5):
    """
    Compare routes matching Colab logic
    """
    # Find connecting routes
    connecting_routes = find_connecting_routes(
        origin_lat, origin_lon, dest_lat, dest_lon,
        routes_df, max_search_distance
    )
    
    if not connecting_routes:
        return pd.DataFrame(), []
    
    # Predict metrics for each route
    route_predictions = []
    
    for route_info in connecting_routes:
        metrics = predict_route_metrics(route_info, departure_time)
        
        arrival_time = departure_time + timedelta(minutes=metrics['predicted_time_min'])
        
        route_predictions.append({
            'route_id': route_info['route_id'],
            'route_name': route_info['route_name'],
            'total_distance_km': metrics['total_distance_km'],
            'predicted_time_min': metrics['predicted_time_min'],
            'avg_speed_kmh': metrics['avg_speed_kmh'],
            'slowest_segment_speed': metrics['slowest_segment_speed'],
            'num_segments': metrics['num_segments'],
            'distance_from_origin_m': route_info['distance_from_origin'] * 1000,
            'distance_from_dest_m': route_info['distance_from_dest'] * 1000,
            'arrival_time': arrival_time.strftime('%H:%M'),
            'coordinates': route_info['coordinates']
        })
    
    if not route_predictions:
        return pd.DataFrame(), []
    
    # Create results dataframe
    results = pd.DataFrame(route_predictions)
    
    # Sort by predicted time (fastest first)
    results = results.sort_values('predicted_time_min').reset_index(drop=True)
    
    # Add ranking
    results['rank'] = range(len(results))
    
    # Calculate time differences
    fastest_time = results['predicted_time_min'].min()
    results['time_diff_vs_fastest_min'] = results['predicted_time_min'] - fastest_time
    
    # Return top N and route IDs
    results_top = results.head(top_n)
    route_ids = results_top['route_id'].tolist()
    
    return results_top, route_ids

@st.cache_data
def load_data():
    try:
        traffic_url = "https://huggingface.co/datasets/Ayemm/BKK_Bus_Data/resolve/main/traffic.csv"
        congestion_url = "https://huggingface.co/datasets/Ayemm/BKK_Bus_Data/resolve/main/congestion_zones.csv"
        bus_routes_url = "https://huggingface.co/datasets/Ayemm/BKK_Bus_Data/resolve/main/cleaned_bus_routes_file.csv"

        traffic_df = pd.read_csv(traffic_url)
        congestion_df = pd.read_csv(congestion_url)
        bus_routes_df = pd.read_csv(bus_routes_url)
        return traffic_df, congestion_df, bus_routes_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def visualize_route_map(routes_comparison_df, origin, destination):
    """Create map with multiple routes"""
    m = folium.Map(location=[13.7563, 100.5018], zoom_start=12)
    
    colors = ['green', 'darkgreen', 'lightgreen', 'orange', 'red']
    
    for idx, row in routes_comparison_df.iterrows():
        coords = row.get('coordinates', [])
        
        if coords and len(coords) > 0:
            color = colors[idx % len(colors)]
            # Convert [lon, lat] to [lat, lon] for folium
            folium_coords = [[lat, lon] for lon, lat in coords]
            folium.PolyLine(
                folium_coords, 
                color=color, 
                weight=4, 
                opacity=0.7,
                popup=f"Route {row['route_id']}: {row['predicted_time_min']:.1f} min"
            ).add_to(m)
    
    # Add origin marker (red house icon)
    if origin:
        folium.Marker(
            origin,
            icon=folium.Icon(color='red', icon='home'),
            popup='Origin'
        ).add_to(m)
    
    # Add destination marker (green flag)
    if destination:
        folium.Marker(
            destination,
            icon=folium.Icon(color='green', icon='flag'),
            popup='Destination'
        ).add_to(m)
    
    return m

# ========== MAIN APP ==========
st.markdown("""
<div class="main-header">
    <h1>Bus Route Optimizer</h1>
    <p>Compare routes between bus stops and find the fastest path</p>
</div>
""", unsafe_allow_html=True)

# Load data
with st.spinner("Loading data..."):
    traffic_df, congestion_df, bus_routes_df = load_data()
    
if traffic_df is None or bus_routes_df is None:
    st.error("Failed to load data. Please check your connection.")
    st.stop()

# ========== INPUT SECTION ==========
st.markdown('<div class="section-header">📍 Select Stops</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.text("Search Origin")
    origin_search = st.text_input("Type to search origin stop", placeholder="Type to search origin stop", label_visibility="collapsed", key="origin_search")
    origin_stop = st.selectbox(
        "Origin Stop",
        ["Elephant Building (13.8247, 100.5667)", "Central World (13.7469, 100.5397)", "Siam Paragon (13.7465, 100.5348)"],
        label_visibility="collapsed"
    )

with col2:
    st.text("Search Destination")
    dest_search = st.text_input("Type to search destination stop", placeholder="Type to search destination stop", label_visibility="collapsed", key="dest_search")
    dest_stop = st.selectbox(
        "Destination Stop",
        ["JJ mall (13.8007, 100.5482)", "Chatuchak Park (13.8028, 100.5496)", "Mo Chit (13.8026, 100.5538)"],
        label_visibility="collapsed"
    )

# Parse coordinates from selection
origin_coords = [float(x.strip()) for x in origin_stop.split('(')[1].split(')')[0].split(',')]
dest_coords = [float(x.strip()) for x in dest_stop.split('(')[1].split(')')[0].split(',')]

# ========== DEPARTURE TIME ==========
st.markdown('<div class="section-header">🕐 Departure Time</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    time_preset = st.selectbox(
        "Time Preset",
        ["Morning Rush (7:00-9:00)", "Midday (11:00-14:00)", "Evening Rush (17:00-19:00)", "Night (20:00-23:00)"]
    )

with col2:
    hour = st.slider("Hour", 0, 23, 8)

with col3:
    minute = st.slider("Minute", 0, 59, 0)

departure_date = st.date_input("Date", datetime.now())

# Create departure datetime
departure_time = datetime.combine(departure_date, time(hour, minute))

# ========== SETTINGS ==========
with st.expander("⚙️ Settings"):
    max_search_distance = st.slider("Maximum search distance (km)", 0.1, 2.0, 0.5, 0.1)
    max_routes = st.slider("Maximum routes to compare", 2, 10, 5)

# ========== RUN COMPARISON ==========
if st.button("🔍 Find Routes", type="primary"):
    with st.spinner("Analyzing routes..."):
        comparison_df, route_ids = compare_routes_streamlit(
            origin_coords[0], origin_coords[1],
            dest_coords[0], dest_coords[1],
            departure_time,
            bus_routes_df,
            max_search_distance=max_search_distance,
            top_n=max_routes
        )
        
        if comparison_df.empty:
            st.error(f"No routes found connecting these points within {max_search_distance} km. Try increasing the search distance.")
        else:
            st.success(f"Found {len(comparison_df)} routes!")
            
            # Store in session state
            st.session_state['comparison_df'] = comparison_df
            st.session_state['route_ids'] = route_ids
            st.session_state['origin_coords'] = origin_coords
            st.session_state['dest_coords'] = dest_coords

# ========== DISPLAY RESULTS ==========
if 'comparison_df' in st.session_state and not st.session_state['comparison_df'].empty:
    comparison_df = st.session_state['comparison_df']
    
    # ========== ROUTE COMPARISON TABLE ==========
    st.markdown('<div class="section-header">📊 Detailed Route Comparison</div>', unsafe_allow_html=True)
    
    # Prepare display dataframe
    display_df = comparison_df[[
        'rank', 'route_id', 'route_name', 'total_distance_km',
        'predicted_time_min', 'avg_speed_kmh', 'time_diff_vs_fastest_min', 'arrival_time'
    ]].copy()
    
    # Format time difference
    display_df['time_diff_vs_fastest_min'] = display_df['time_diff_vs_fastest_min'].apply(
        lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}"
    )
    
    # Style the dataframe
    def style_comparison(df):
        def highlight_time(row):
            colors = [''] * len(row)
            time_col_idx = df.columns.get_loc('predicted_time_min')
            
            if row['predicted_time_min'] == df['predicted_time_min'].min():
                colors[time_col_idx] = 'background-color: #90EE90'  # Light green
            elif row['predicted_time_min'] == df['predicted_time_min'].max():
                colors[time_col_idx] = 'background-color: #FFB6C6'  # Light red
            
            return colors
        
        return df.style.apply(highlight_time, axis=1)
    
    styled_df = style_comparison(display_df)
    st.dataframe(styled_df, use_container_width=True, height=min(250, 50 + 50 * len(display_df)))
    
    # ========== ROUTE MAP ==========
    st.markdown('<div class="section-header">🗺️ Route Map</div>', unsafe_allow_html=True)
    
    route_map = visualize_route_map(
        comparison_df,
        origin=st.session_state['origin_coords'],
        destination=st.session_state['dest_coords']
    )
    folium_static(route_map, width=1200, height=500)
    
    # ========== SUMMARY METRICS ==========
    st.markdown('<div class="section-header">📈 Route Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    best_route = comparison_df.iloc[0]
    
    with col1:
        st.metric("Total Routes Found", len(comparison_df))
    
    with col2:
        st.metric("Fastest Route", f"{best_route['route_name']}")
    
    with col3:
        st.metric("Fastest Time", f"{best_route['predicted_time_min']:.1f} min")
    
    with col4:
        if len(comparison_df) > 1:
            time_saved = comparison_df.iloc[-1]['time_diff_vs_fastest_min']
            st.metric("Time Saved", f"{time_saved:.1f} min", delta=f"-{time_saved:.1f}")
        else:
            st.metric("Time Saved", "N/A")
    
    # ========== RECOMMENDATION ==========
    st.markdown('<div class="section-header">💡 Recommendation</div>', unsafe_allow_html=True)
    
    st.success(f"""
    **Take Route {best_route['route_id']} ({best_route['route_name']})**
    
    - 🚀 Departure: {departure_time.strftime('%H:%M')}
    - 🎯 Expected Arrival: {best_route['arrival_time']}
    - ⏱️ Total Journey: {best_route['predicted_time_min']:.0f} minutes
    - 📏 Distance: {best_route['total_distance_km']:.2f} km
    - 🚶 Walking from origin: {best_route['distance_from_origin_m']:.0f} m
    """)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("*Predictions based on historical traffic patterns and time of day*")