import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# Generate synthetic dataset for stars with more realistic ranges
np.random.seed(0)
data_size = 500
data = {
    'Mass': np.random.uniform(1500, 5000, data_size),       # Mass range in solar masses
    'Temperature': np.random.uniform(2.0, 7.0, data_size),  # Temperature range in Kelvin
    'Radius': np.random.uniform(100, 500, data_size)        # Radius range in solar radii
}
df = pd.DataFrame(data)

X = df[['Mass', 'Temperature', 'Radius']]   # labels are optional for unsupervised

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Plotting the clusters (using Mass and Radius for 2D visualization)
plt.scatter(df['Mass'], df['Radius'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Mass (solar masses)')
plt.ylabel('Radius (solar radii)')
plt.title('Star Clusters based on Mass and Radius')
plt.show()
