

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
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px
import plotly.io as pio
import requests
from io import StringIO
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import plotly.graph_objects as go
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
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

#%% 🧹 Clean Data for Clustering

# Drop non-numeric or irrelevant columns
df_prem_clean = df_prem.drop(columns=['Rk', 'Squad', 'MP', 'Pts/MP', 'GD', 'xGD/90', 'Top Team Scorer', 'Goalkeeper', 'Notes'], errors='ignore')

# Apply Z-Score Standardization
df_prem_scaled = df_prem_clean.apply(zscore, ddof=1)

# Verify standardization
print("Means after standardization:\n", round(df_prem_scaled.mean(), 2))
print("Standard deviations after standardization:\n", round(df_prem_scaled.std(), 2))

#%% Visualizing information about the data and variables

# Descriptive statistics of the variables

tab_desc = df_prem.describe()

# Correlation matrix of the variables

# Generating Pearson correlation matrix
correlation_matrix = pg.rcorr(df_prem_scaled, method='pearson', upper='pval',
                              decimals=4,
                              pval_stars={0.01: '***', 0.05: '**', 0.10: '*'})

#%% Heatmap showing the correlation between attributes

# Basic correlation matrix
corr = df_prem_scaled.corr()

# Heatmap plot
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x=corr.columns,
        y=corr.index,
        z=np.array(corr),
        text=corr.values,
        texttemplate='%{text:.2f}',
        colorscale='viridis'))

fig.update_layout(
    height=600,
    width=600)

fig.show()

#%% Bartlett's Sphericity Test

bartlett, p_value = calculate_bartlett_sphericity(df_prem_scaled[['W','D','L','GF','GA','Pts','xG','xGA','xGD','Attendance']])

print(f'Bartlett Chi-square: {round(bartlett, 2)}')
print(f'p-value: {round(p_value, 4)}')

#%% Defining PCA (initial procedure with all possible factors)

fa = FactorAnalyzer(n_factors=10, method='principal', rotation='varimax').fit(df_prem_scaled)

#%% Obtaining eigenvalues: results from FactorAnalyzer function

eigenvalues = fa.get_eigenvalues()[0]

print(eigenvalues)

#%% Redefining PCA by latent root criterion

# There are two factors with eigenvalues > 1

fa = FactorAnalyzer(n_factors=2, method='principal', rotation=None).fit(df_prem_scaled)

#%% Eigenvalues, variances and cumulative variances

factor_eigenvalues = fa.get_factor_variance()

eigen_table = pd.DataFrame(factor_eigenvalues)
eigen_table.columns = [f"Factor {i+1}" for i, v in enumerate(eigen_table.columns)]
eigen_table.index = ['Eigenvalue', 'Variance', 'Cumulative Variance']
eigen_table = eigen_table.T

print(eigen_table)

#%% Determining factor loadings

factor_loadings = fa.loadings_

loadings_table = pd.DataFrame(factor_loadings)
loadings_table.columns = [f"Factor {i+1}" for i, v in enumerate(loadings_table.columns)]
loadings_table.index = df_prem_scaled.columns

print(loadings_table)

#%% Loading plot of factor loadings

plt.figure(figsize=(12,8))
loadings_chart = loadings_table.reset_index()
plt.scatter(loadings_chart['Factor 1'], loadings_chart['Factor 2'], s=30, color='red')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.02, point['y'], point['val'], fontsize=8)

label_point(x=loadings_chart['Factor 1'],
            y=loadings_chart['Factor 2'],
            val=loadings_chart['index'],
            ax=plt.gca()) 

plt.axhline(y=0, color='grey', ls='--')
plt.axvline(x=0, color='grey', ls='--')
plt.ylim([-1.1,1.1])
plt.xlim([-1.1,1.1])
plt.title("Loading Plot", fontsize=16)
plt.xlabel(f"Factor 1: {round(eigen_table.iloc[0]['Variance']*100,2)}% variance explained", fontsize=12)
plt.ylabel(f"Factor 2: {round(eigen_table.iloc[1]['Variance']*100,2)}% variance explained", fontsize=12)
plt.show()

#%% Determining communalities

communalities = fa.get_communalities()

communalities_table = pd.DataFrame(communalities)
communalities_table.columns = ['Communalities']
communalities_table.index = df_prem_scaled.columns

print(communalities_table)

#%% Factor extraction for observations in the dataset

factors = pd.DataFrame(fa.transform(df_prem_scaled))
factors.columns = [f"Factor {i+1}" for i, v in enumerate(factors.columns)]

#%% Identifying factor scores

scores = fa.weights_

scores_table = pd.DataFrame(scores)
scores_table.columns = [f"Factor {i+1}" for i, v in enumerate(scores_table.columns)]
scores_table.index = df_prem_scaled.columns

print(scores_table)

#%% Interpreting factor scores in each extracted factor

scores_graph = scores_table.reset_index()
scores_graph = scores_graph.melt(id_vars='index')

sns.barplot(data=scores_graph, x='variable', y='value', hue='index', palette='viridis')
plt.legend(title='Variables', bbox_to_anchor=(1,1), fontsize='6')
plt.title('Factor Scores', fontsize='12')
plt.xlabel(xlabel=None)
plt.ylabel(ylabel=None)
plt.show()

#%% The factors remain orthogonal

pg.rcorr(factors[['Factor 1', 'Factor 2']], 
         method='pearson', upper='pval', 
         decimals=4, 
         pval_stars={0.01: '***', 0.05: '**', 0.10: '*'})

#%% End!

#%% 🌳 Hierarchical Clustering on PCA Factors
# Compute distances using factor scores
euclidean_dist_pca = pdist(factors, metric='euclidean')

# Dendrogram plot using PCA scores
plt.figure(figsize=(16, 8))
dend_pca = sch.linkage(factors, method='complete', metric='euclidean')
sch.dendrogram(dend_pca, color_threshold=2, labels=list(df_prem['Squad']))
plt.title('Dendrogram (PCA Scores)', fontsize=16)
plt.xlabel('Teams', fontsize=14)
plt.ylabel('Euclidean Distance', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.axhline(y=2, color='red', linestyle='--')
plt.tight_layout()
plt.show()

#%% 🔗 Apply Hierarchical Clustering on PCA Scores
hier_pca = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='complete')
df_prem['Cluster_Hierarchical_PCA'] = hier_pca.fit_predict(factors)

# View clusters
for cluster in sorted(df_prem['Cluster_Hierarchical_PCA'].unique()):
    print(f"\n🔷 Cluster {cluster} (PCA):")
    teams = df_prem[df_prem['Cluster_Hierarchical_PCA'] == cluster]['Squad'].values
    print(", ".join(teams))

#%% 📈 Elbow Method on PCA Scores
elbow_pca = []
K = range(1, 6)
for k in K:
    model = KMeans(n_clusters=k, init='random', random_state=100).fit(factors)
    elbow_pca.append(model.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, elbow_pca, marker='o')
plt.xlabel('Number of Clusters', fontsize=14)
plt.ylabel('WCSS (Inertia)', fontsize=14)
plt.title('Elbow Method (PCA)', fontsize=16)
plt.xticks(K)
plt.show()

#%% 🟪 Silhouette Score on PCA Factors
silhouette_pca = []
K = range(2, 6)
for k in K:
    model = KMeans(n_clusters=k, init='random', random_state=100).fit(factors)
    score = silhouette_score(factors, model.labels_)
    silhouette_pca.append(score)

plt.figure(figsize=(10,6))
plt.plot(K, silhouette_pca, color='purple', marker='o')
plt.xlabel('Number of Clusters', fontsize=14)
plt.ylabel('Silhouette Score', fontsize=14)
plt.title('Silhouette Method (PCA)', fontsize=16)
plt.axvline(x=silhouette_pca.index(max(silhouette_pca)) + 2, color='red', linestyle='--')
plt.show()

#%% 🔐 Final KMeans Clustering on PCA Scores (e.g., 5 Clusters)
kmeans_pca = KMeans(n_clusters=5, init='random', random_state=100)
df_prem['Cluster_KMeans_PCA'] = kmeans_pca.fit_predict(factors)

# Show teams in each cluster
for cluster in sorted(df_prem['Cluster_KMeans_PCA'].unique()):
    print(f"\n🔷 Cluster {cluster} (KMeans - PCA):")
    teams = df_prem[df_prem['Cluster_KMeans_PCA'] == cluster]['Squad'].values
    print(", ".join(teams))

#%% 📊 PCA Scatter Plot – Teams in Factor Space (Factor 1 vs Factor 2)

# Add team names from original dataset
factors.index = df_prem['Squad']

# Plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid", font_scale=1.1)

# Scatterplot
sns.scatterplot(data=factors, x='Factor 1', y='Factor 2', s=100, color='dodgerblue')

# Annotate team names
for team in factors.index:
    plt.text(x=factors.loc[team, 'Factor 1'] + 0.02, 
             y=factors.loc[team, 'Factor 2'], 
             s=team, fontsize=9)

# Reference lines
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')

# Labels and title
plt.title('Teams in PCA Factor Space (Factor 1 vs Factor 2)', fontsize=16)
plt.xlabel(f"Factor 1: {round(eigen_table.iloc[0]['Variance']*100, 2)}% variance explained", fontsize=12)
plt.ylabel(f"Factor 2: {round(eigen_table.iloc[1]['Variance']*100, 2)}% variance explained", fontsize=12)

plt.tight_layout()
plt.show()
