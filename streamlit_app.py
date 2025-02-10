import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Load dataset (Replace with actual file)
df = pd.read_csv("sl1.csv")

# Data Preprocessing
df.dropna(inplace=True)
scaler = StandardScaler()
df[['Pressure', 'P1', 'P2', 'Orifice Size']] = scaler.fit_transform(df[['Pressure (Barg)', 'P1 (PSIA)', 'P2 (PSIA)', 'Orifice Size D (IN INCHES)']])

# Streamlit App Layout
st.title("Steam Loss Analysis Dashboard")
st.sidebar.header("Select Visualization")
option = st.sidebar.selectbox("Choose an analysis", ["Heatmap - Steam Loss by Location", "Failure Type Distribution", "Historical Trends", "Anomaly Detection", "Predictive Model", "Clustering Analysis"])

# Heatmap Visualization
if option == "Heatmap - Steam Loss by Location":
    fig = px.density_mapbox(df, lat='Latitude', lon='Longitude', z='Loss in KG/HR', radius=10,
                            mapbox_style="stamen-terrain", title="Steam Loss Distribution")
    st.plotly_chart(fig)

# Pie Chart - Failure Type Distribution
elif option == "Failure Type Distribution":
    fig, ax = plt.subplots()
    df['Type of Failure'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=["#ff9999", "#66b3ff"])
    ax.set_ylabel('')
    st.pyplot(fig)

# Time-Series Analysis
elif option == "Historical Trends":
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    fig, ax = plt.subplots()
    df['Loss in KG/HR'].plot(ax=ax, color='blue', title="Historical Steam Loss Trends")
    st.pyplot(fig)

# Anomaly Detection
elif option == "Anomaly Detection":
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = iso_forest.fit_predict(df[['Loss in KG/HR']])
    anomaly_df = df[df['Anomaly'] == -1]
    fig = px.scatter(anomaly_df, x='Timestamp', y='Loss in KG/HR', color_discrete_sequence=['red'], title="Detected Anomalies")
    st.plotly_chart(fig)

# Predictive Model
elif option == "Predictive Model":
    X = df[['Pressure', 'P1', 'P2', 'Orifice Size']]
    y = df['Loss in KG/HR']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    df['Predicted Loss'] = model.predict(X)
    fig = px.line(df, x=df.index, y=['Loss in KG/HR', 'Predicted Loss'], title="Steam Loss Prediction")
    st.plotly_chart(fig)

# Clustering Analysis
elif option == "Clustering Analysis":
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Pressure', 'P1', 'P2', 'Orifice Size']])
    fig = px.scatter(df, x='Pressure', y='Loss in KG/HR', color='Cluster', title="Steam Loss Clustering Analysis")
    st.plotly_chart(fig)

st.sidebar.info("This dashboard provides insights into steam loss trends, anomaly detection, and predictive analytics.")
