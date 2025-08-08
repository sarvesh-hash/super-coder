# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load Data
df = pd.read_csv('Traffic_V2.csv')

print("First few rows :\n",df.head())
print("Data Description : \n",df.describe())

print("Missing Values : ")
print(df.isnull().sum())

# Data Preprocessing
# Convert Date and Time
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Time'] = pd.to_datetime(df['Time'], format='%I.%M.%S %p').dt.time
df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Feature Engineering
df['Hour'] = df['Datetime'].dt.hour
df['Day'] = df['Datetime'].dt.day
df['Month'] = df['Datetime'].dt.month
df['Traffic_Density'] = df['Vehicle_Count'] / (df['Vehicle_Speed'] + 1)

# Descriptive Statistics
print("\n Summary Statistics:")
print(df[['Vehicle_Count', 'Vehicle_Speed', 'Congestion_Level']].describe())

# Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.countplot(x='Congestion_Level', data=df)
plt.title('Congestion Level Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Peak_Off_Peak', y='Vehicle_Speed', data=df)
plt.title('Vehicle Speed During Peak vs Off-Peak')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Hour', y='Vehicle_Count', data=df)
plt.title('Hourly Vehicle Count')
plt.show()

# Correlation Analysis
corr = df[['Vehicle_Count', 'Vehicle_Speed', 'Congestion_Level']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Sensor Location Congestion Analysis
location_congestion = df.groupby('Location')['Congestion_Level'].mean().sort_values(ascending=False)
print("\n Average Congestion by Sensor Location:")
print(location_congestion)


# Predicting Congestion Level (Classification)
X = df[['Vehicle_Count', 'Vehicle_Speed', 'Hour', 'Traffic_Density']]
y = df['Congestion_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n Congestion Level Prediction Report:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Predicting Future Vehicle Count (Regression)
X_reg = df[['Vehicle_Speed', 'Hour', 'Traffic_Density']]
y_reg = df['Target_Vehicle_Count']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

print("\n Vehicle Count Prediction Performance:")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.2f}")
print(f"RMSE: {mean_squared_error(y_test_reg, y_pred_reg)**0.5:.2f}")

# Prepare Data for Clustering
cluster_features = df[['Vehicle_Count', 'Vehicle_Speed', 'Congestion_Level', 'Hour']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

#Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize Clusters with PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)
df['PC1'] = pca_components[:, 0]
df['PC2'] = pca_components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='Set2')
plt.title('Traffic Clusters (PCA Projection)')
plt.show()

# Cluster Analysis by Location
location_cluster = df.groupby(['Location', 'Cluster']).size().unstack(fill_value=0)
print("\n Cluster Distribution by Sensor Location:")
print(location_cluster)

# Cluster-wise Traffic Behavior
cluster_summary = df.groupby('Cluster')[['Vehicle_Count', 'Vehicle_Speed', 'Congestion_Level']].mean()
print("\n Average Metrics per Cluster:")
print(cluster_summary)

# Correlation analysis
corr_matrix = df[['Vehicle_Count','Vehicle_Speed','Congestion_Level']].corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()