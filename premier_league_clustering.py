
#%% ✅ Install Required Packages (Run if not installed)
# Uncomment this if running in a new environment
# !pip install pandas numpy matplotlib seaborn plotly scikit-learn pingouin lxml requests

#%% 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering, KMeans
import pingouin as pg
import plotly.express as px
import plotly.io as pio
import requests
from io import StringIO
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
print(os.getcwd())
pio.renderers.default = 'browser'

#%% 🌐 Function to Fetch Team Stats from FBref

def fetch_fbref_table(url, table_name='Squad'):
    """
    Fetches table from FBref by URL and table name.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
        df_list = pd.read_html(StringIO(html))

        for df in df_list:
            if table_name in df.columns:
                return df

        raise ValueError(f"Table '{table_name}' not found at {url}")

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

#%% 🔗 Fetch Premier League Data (2024/2025 Example)

url_prem = "https://fbref.com/en/comps/9/Premier-League-Stats"
df_prem = fetch_fbref_table(url_prem, table_name='Squad')

if df_prem is None:
    raise Exception("Premier League data unavailable. Check the URL or table name.")

#%% 🔍 Check Dataset Structure

print(df_prem.info())
print(df_prem.describe())

#%% 📊 Quick 3D Plot of Example Stats

fig = px.scatter_3d(df_prem, 
                     x='xG', 
                     y='xGA', 
                     z='xGD',
                     text=df_prem['Squad'])
fig.show()

#%% 🧹 Clean Data for Clustering

# Drop non-numeric or irrelevant columns
df_prem_clean = df_prem.drop(columns=['Rk', 'Squad', 'MP', 'Top Team Scorer', 'Goalkeeper', 'Notes'], errors='ignore')

# Apply Z-Score Standardization
df_prem_scaled = df_prem_clean.apply(zscore, ddof=1)

# Verify standardization
print("Means after standardization:\n", round(df_prem_scaled.mean(), 2))
print("Standard deviations after standardization:\n", round(df_prem_scaled.std(), 2))

#%% 🌳 Hierarchical Clustering - Dendrogram

# Compute pairwise Euclidean distances
euclidean_dist = pdist(df_prem_scaled, metric='euclidean')

# Plot dendrogram (Single Linkage example)
plt.figure(figsize=(16, 8))
dend = sch.linkage(df_prem_scaled, method='complete', metric='euclidean')
sch.dendrogram(dend, color_threshold=4, labels=list(df_prem['Squad']))
plt.title('Dendrogram - Complete Linkage', fontsize=16)
plt.xlabel('Teams', fontsize=14)
plt.ylabel('Euclidean Distance', fontsize=14)
plt.xticks(rotation=90, fontsize=10)  # Rotate team labels vertically
plt.axhline(y=4, color='red', linestyle='--')
plt.tight_layout()
plt.show()

#%% 🔗 Apply Hierarchical Clustering

hierarchical = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='complete')
df_prem['Cluster_Hierarchical'] = hierarchical.fit_predict(df_prem_scaled)

# Print teams in each cluster
for cluster in sorted(df_prem['Cluster_Hierarchical'].unique()):
    print(f"\n🔷 Cluster {cluster}:")
    teams = df_prem[df_prem['Cluster_Hierarchical'] == cluster]['Squad'].values
    print(", ".join(teams))

#%% 📈 Elbow Method for K-Means

elbow = []
K = range(1, 6)
for k in K:
    model = KMeans(n_clusters=k, init='random', random_state=100).fit(df_prem_scaled)
    elbow.append(model.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, elbow, marker='o')
plt.xlabel('Number of Clusters', fontsize=14)
plt.ylabel('WCSS (Inertia)', fontsize=14)
plt.title('Elbow Method for Optimal K', fontsize=16)
plt.xticks(K)
plt.show()

#%% 🔐 Apply K-Means Clustering (e.g., 3 Clusters)

kmeans = KMeans(n_clusters=4, init='random', random_state=100)
df_prem['Cluster_KMeans'] = kmeans.fit_predict(df_prem_scaled)

#%% 📍 Inspect K-Means Cluster Centroids

centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df_prem_scaled.columns)
centroids.index.name = 'Cluster'
print(centroids)

#%% 🎯 2D Cluster Plot Example

plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_prem, 
                x='W', 
                y='xG', 
                hue='Cluster_KMeans', 
                palette='viridis', 
                s=100)
plt.title('K-Means Clusters', fontsize=16)
plt.xlabel('Wins (W)', fontsize=14)
plt.ylabel('Expected Goals (xG)', fontsize=14)
plt.legend()
plt.show()

#%% 🧠 3D Cluster Plot

fig = px.scatter_3d(df_prem, 
                     x='W', 
                     y='xG', 
                     z='L',
                     color='Cluster_KMeans',
                     text=df_prem['Squad'])
fig.show()

import plotly.graph_objects as go


# Create the scatter points
scatter = go.Scatter3d(
    x=df_prem['W'],
    y=df_prem['xG'],
    z=df_prem['L'],
    mode='markers+text',
    text=df_prem['Squad'],
    textposition='top center',
    marker=dict(
        size=6,
        color=df_prem['Cluster_KMeans'],
        colorscale='Viridis',
        opacity=0.8,
        line=dict(width=0.5, color='darkgrey')
    )
)

# Create the figure
fig = go.Figure(data=[scatter])

# Layout settings for better readability
fig.update_layout(
    title='3D Cluster Plot with Team Labels',
    scene=dict(
        xaxis_title='Wins (W)',
        yaxis_title='Expected Goals (xG)',
        zaxis_title='Losses (L)'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()


#%% 🧪 ANOVA Test - Cluster Differences

for var in ['W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Pts/MP', 'xG', 'xGA', 'xGD', 'xGD/90', 'Attendance']:
    print(f"\nANOVA for {var}:")
    display(
        pg.anova(dv=var, between='Cluster_KMeans', data=df_prem, detailed=True).T
    )

#%% 🔥 Heatmap of Clusters

df_heatmap = df_prem_scaled.copy()
df_heatmap['Cluster_KMeans'] = df_prem['Cluster_KMeans']
df_heatmap['Squad'] = df_prem['Squad']

# Sort by cluster label
df_heatmap = df_heatmap.sort_values('Cluster_KMeans')

plt.figure(figsize=(12, 8))
sns.heatmap(df_heatmap.drop(columns=['Cluster_KMeans', 'Squad']),
            cmap='viridis',
            yticklabels=df_heatmap['Squad'],
            cbar=True)
plt.title('Cluster Heatmap of Premier League Teams', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Teams', fontsize=14)
plt.show()

#%% ⚽️ Scatter Plot with Club Logos Function

def plot_with_logos(df, x_col, y_col, cluster_col=None, title="Cluster Plot with Logos"):
    """
    Plot scatter with club logos instead of points.
    Logos should be saved in './logos/{Team}.png'.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Base invisible scatter for axis scaling
    ax.scatter(df[x_col], df[y_col], s=0)

    for _, row in df.iterrows():
        try:
            path = f'./logos/{row["Squad"]}.png'
            logo = Image.open(path)
            im = OffsetImage(logo, zoom=0.12)
            ab = AnnotationBbox(im, (row[x_col], row[y_col]), frameon=False)
            ax.add_artist(ab)
        except FileNotFoundError:
            ax.text(row[x_col], row[y_col], row["Squad"], fontsize=8, ha='center')

    if cluster_col:
        for cluster in sorted(df[cluster_col].unique()):
            clustered = df[df[cluster_col] == cluster]
            ax.scatter(clustered[x_col], clustered[y_col], label=f'Cluster {cluster}', s=0)

        ax.legend()

    ax.set_xlabel(x_col, fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

#%% 🎨 Plot Example - Logos Scatter

plot_with_logos(
    df=df_prem,
    x_col='xG',
    y_col='xGA',
    cluster_col='Cluster_KMeans',
    title='Premier League Clusters - xG vs xGA with Logos'
)

#%% Radar Charts for Each Cluster (Premier League)

import matplotlib.pyplot as plt
import numpy as np

# Select only the standardized features for plotting
features = df_prem_scaled.columns.tolist()

# Combine standardized data with clusters and team names
df_plot = df_prem_scaled.copy()
df_plot['Cluster_KMeans'] = df_prem['Cluster_KMeans']
df_plot['Squad'] = df_prem['Squad']

# Number of features
N = len(features)

# Angle for each axis in the plot
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

# Plot radar chart per cluster
for cluster in sorted(df_plot['Cluster_KMeans'].unique()):
    subset = df_plot[df_plot['Cluster_KMeans'] == cluster]
    
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)

    for idx, row in subset.iterrows():
        values = row[features].tolist()
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, label=row['Squad'], linewidth=1)
        ax.fill(angles, values, alpha=0.05)

    ax.set_title(f'Cluster {cluster} - Premier League', size=16, y=1.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=10)
    ax.set_yticklabels([])  # Remove radial labels for clarity
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()
