import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('Iris.csv')
print(df.head())

'''preprocessing of Species'''
number = LabelEncoder()
df['Species'] = number.fit_transform(df['Species'].astype('str'))


'''visualisation to check the clusters'''
xs = df['SepalLengthCm']
ys = df['SepalWidthCm']
plt.scatter(xs,ys,c=df['Species'])
plt.show()

'''Kmeans for clustering'''
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
model = KMeans(n_clusters = 3)
model = model.fit(X)
print(model)
labels = model.predict(X)


'''Scatter plot with centroids'''
plt.scatter(xs,ys,c=labels,alpha=0.5)
centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
plt.scatter(centroids_x,centroids_y,marker='D',s=50)
plt.show()

'''evaluating the clusters'''

df_eval = pd.DataFrame({'labels':labels,'species':df['Species']})
ct = pd.crosstab(df_eval['labels'],df_eval['species'])
print(ct)

print(model.inertia_)
#plotting inertias vs number of clusters
ks = range(1,6)
inertias = []
for k in ks:
    model = KMeans(n_clusters = k)
    model.fit(X)
    inertias.append(model.inertia_)
plt.plot(ks,inertias,'-o')
plt.xlabel('number of clusters k')
plt.ylabel('inertias')
plt.xticks(ks)
plt.show()
