import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

print("Generating dummy traffic data...")

# ============================================================================
# GENERATE TRAFFIC DATA
# ============================================================================

# Parameters
n_records = 50000
n_vehicles = 500
start_date = datetime(2024, 10, 1)
end_date = datetime(2024, 10, 7)

# Bangkok coordinates (roughly)
bangkok_lat_min, bangkok_lat_max = 13.65, 13.85
bangkok_lon_min, bangkok_lon_max = 100.45, 100.65

# Generate base data
vehicle_ids = np.random.choice(range(1000, 1000 + n_vehicles), n_records)
timestamps = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) 
              for _ in range(n_records)]

lats = np.random.uniform(bangkok_lat_min, bangkok_lat_max, n_records)
lons = np.random.uniform(bangkok_lon_min, bangkok_lon_max, n_records)

# Create traffic dataframe
traffic_df = pd.DataFrame({
    'Unnamed: 0': range(n_records),
    'VehicleID': vehicle_ids,
    'gpsvalid': np.random.choice([0, 1], n_records, p=[0.05, 0.95]),
    'lat': lats,
    'lon': lons,
    'timestamp': timestamps,
    'heading': np.random.uniform(0, 360, n_records),
    'for_hire_light': np.random.choice([0, 1], n_records, p=[0.7, 0.3]),
    'engine_acc': np.random.choice([0, 1], n_records, p=[0.3, 0.7])
})

# Sort by timestamp
traffic_df = traffic_df.sort_values('timestamp').reset_index(drop=True)

# Add time-based features
traffic_df['hour'] = traffic_df['timestamp'].dt.hour
traffic_df['day_of_week'] = traffic_df['timestamp'].dt.dayofweek
traffic_df['is_weekend'] = traffic_df['day_of_week'].isin([5, 6]).astype(int)
traffic_df['is_rush_hour'] = traffic_df['hour'].isin([7, 8, 17, 18, 19]).astype(int)

# Generate realistic speeds based on time and location
base_speed = 45
rush_hour_penalty = -15
weekend_bonus = 10
night_bonus = 15

traffic_df['speed'] = base_speed + np.random.normal(0, 15, n_records)
traffic_df.loc[traffic_df['is_rush_hour'] == 1, 'speed'] += rush_hour_penalty
traffic_df.loc[traffic_df['is_weekend'] == 1, 'speed'] += weekend_bonus
traffic_df.loc[traffic_df['hour'].isin([0, 1, 2, 3, 4, 5]), 'speed'] += night_bonus

# Ensure speeds are reasonable
traffic_df['speed'] = traffic_df['speed'].clip(0, 120)

# Add location grid
traffic_df['lat_grid'] = (traffic_df['lat'] * 100).round()
traffic_df['lon_grid'] = (traffic_df['lon'] * 100).round()
traffic_df['location_id'] = traffic_df['lat_grid'].astype(str) + '_' + traffic_df['lon_grid'].astype(str)

# Add aggregated features
traffic_df['location_avg_speed'] = traffic_df.groupby('location_id')['speed'].transform('mean')
traffic_df['hour_avg_speed'] = traffic_df.groupby('hour')['speed'].transform('mean')

# Distance from Bangkok center
bangkok_center_lat, bangkok_center_lon = 13.7563, 100.5018
traffic_df['distance_from_center'] = np.sqrt(
    (traffic_df['lat'] - bangkok_center_lat)**2 +
    (traffic_df['lon'] - bangkok_center_lon)**2
)

# Placeholder for congestion features (will be added after congestion zones)
traffic_df['near_congestion'] = 0
traffic_df['distance_to_congestion'] = 0.0

print(f"Generated {len(traffic_df):,} traffic records")

# ============================================================================
# GENERATE CONGESTION ZONES
# ============================================================================

n_zones = 50

# Generate congestion zone centers (clustered around certain areas)
congestion_centers = [
    (13.7563, 100.5018),  # City center
    (13.7308, 100.5215),  # Silom
    (13.7467, 100.5351),  # Sukhumvit
    (13.7650, 100.5380),  # Asok
    (13.8000, 100.5500),  # Chatuchak
]

zone_lats = []
zone_lons = []
for _ in range(n_zones):
    center = congestion_centers[np.random.randint(0, len(congestion_centers))]
    zone_lats.append(center[0] + np.random.normal(0, 0.02))
    zone_lons.append(center[1] + np.random.normal(0, 0.02))

congestion_df = pd.DataFrame({
    'Unnamed: 0': range(n_zones),
    'zone_id': range(n_zones),
    'center_lat': zone_lats,
    'center_lon': zone_lons,
    'avg_speed': np.random.uniform(5, 30, n_zones),
    'size': np.random.randint(20, 200, n_zones),
    'severity': np.random.choice(['Critical', 'High', 'Medium', 'Low'], n_zones, 
                                 p=[0.3, 0.3, 0.25, 0.15])
})

print(f"Generated {len(congestion_df):,} congestion zones")

# ============================================================================
# ADD CONGESTION PROXIMITY FEATURES TO TRAFFIC DATA
# ============================================================================
from scipy.spatial import cKDTree

print("Calculating congestion proximity...")

# Build KDTree
congestion_coords = congestion_df[['center_lat', 'center_lon']].values
tree = cKDTree(congestion_coords)

# Query for nearest congestion zone
traffic_coords = traffic_df[['lat', 'lon']].values
distances, indices = tree.query(traffic_coords, k=1)

# Add features
proximity_threshold = 0.01
traffic_df['near_congestion'] = (distances <= proximity_threshold).astype(int)
traffic_df['distance_to_congestion'] = distances

# Add congestion zone info
severity_map = {'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}
congestion_df['severity_encoded'] = congestion_df['severity'].map(severity_map)

traffic_df['congestion_severity_encode'] = congestion_df.iloc[indices]['severity_encoded'].values
traffic_df['congestion_avg_speed'] = congestion_df.iloc[indices]['avg_speed'].values

print(f"Points near congestion: {traffic_df['near_congestion'].sum():,}")

# ============================================================================
# SAVE FILES
# ============================================================================

traffic_df.to_csv('dummy_traffic.csv', index=False)
congestion_df.to_csv('dummy_congestion_zones.csv', index=False)

print("\n" + "="*80)
print("FILES SAVED SUCCESSFULLY!")
print("="*80)
print(f"\n✅ dummy_traffic.csv: {len(traffic_df):,} records")
print(f"   Columns: {list(traffic_df.columns)}")
print(f"\n✅ dummy_congestion_zones.csv: {len(congestion_df):,} zones")
print(f"   Columns: {list(congestion_df.columns)}")

print("\n" + "="*80)
print("SAMPLE DATA PREVIEW")
print("="*80)
print("\nTraffic Data (first 5 rows):")
print(traffic_df.head())
print("\nCongestion Zones (first 5 rows):")
print(congestion_df.head())

print("\n" + "="*80)
print("STATISTICS")
print("="*80)
print(f"\nTraffic Data:")
print(f"  Date range: {traffic_df['timestamp'].min()} to {traffic_df['timestamp'].max()}")
print(f"  Avg speed: {traffic_df['speed'].mean():.2f} km/h")
print(f"  Speed range: {traffic_df['speed'].min():.2f} - {traffic_df['speed'].max():.2f} km/h")
print(f"  Unique vehicles: {traffic_df['VehicleID'].nunique()}")
print(f"  Points near congestion: {traffic_df['near_congestion'].sum():,} ({traffic_df['near_congestion'].mean()*100:.1f}%)")

print(f"\nCongestion Zones:")
print(f"  Severity distribution:")
for severity, count in congestion_df['severity'].value_counts().items():
    print(f"    {severity}: {count}")

print("\n✅ You can now upload these files to your Streamlit dashboard!")