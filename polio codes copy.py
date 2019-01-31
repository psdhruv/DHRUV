import pandas as pd

p= pd.read_csv("polio.csv",index_col=0)
c=p["Health Care  Data"]
plt.hist(c,range=(c.min(),c.max()))
c=c.fillna(c.mean())
p["Health Care  Data"]=c
p["Migration Data"]=p["Migration Data"].fillna(p["Migration Data"].mean())

p["Cleanliness Data"]=p["Cleanliness Data"].fillna(p["Cleanliness Data"].mean())
p["Mean Temp. Data"]=p["Mean Temp. Data"].fillna(p["Mean Temp. Data"].mean())
p["Vaccines to birth"]=p["Vaccines to birth"].fillna(p["Vaccines to birth"].mean())
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans



p1= p[["Health Care  Data","Migration Data","Cleanliness Data","Mean Temp. Data","Vaccines to birth","improved sanitation" ]]

p1= p[["Health Care  Data","Migration Data","Cleanliness Data","Mean Temp. Data","Vaccines to birth" ]]
p1["improved sanitation"]= p["improved sanitation"]
p1["improved sanitation"]= p["improved sanitation"]
p["improved sanitation"]
p1["improved sanitation"]= p["improved sanitation"]
p1["Health Care  Data"]=p1["Health Care  Data"]/p1["Health Care  Data"].max()
p1["Migration Data"]=p1["Migration Data"]/p1["Migration Data"].max()
p1["Cleanliness Data"]=p1["Cleanliness Data"]/p1["Cleanliness Data"].max()
p1["Mean Temp. Data"]=p1["Mean Temp. Data"]/p1["Mean Temp. Data"].max()
p1["Vaccines to birth"]=p1["Vaccines to birth"]/p1["Vaccines to birth"].max()
p1["improved sanitation"]=p1["improved sanitation"]/p1["improved sanitation"].max()


kmeans = KMeans(n_clusters=3)
kmeans.fit(p1)
print(kmeans.labels_)
print(kmeans.labels_)
print(kmeans.labels_)
kmeans = KMeans(n_clusters=3)
kmeans.fit(p1)
print(kmeans.labels_)

z=kmeans.labels_
z=z+1
z

p["label"]=z


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, max_leaf_nodes=10,random_state=42)
rf.fit(p1, p["label"])
predictions = rf.predict(p1)
predictions
p["predicted  level"]=predictions
pd.to_csv("polio results.csv")
