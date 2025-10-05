import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_folium import folium_static
import folium
from math import radians, sin, cos, sqrt, atan2

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Bangkok Bus Route Comparator",
    layout="wide",
)


# need to modify this according to real-time data