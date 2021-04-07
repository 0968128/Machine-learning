from machinelearningdata import Machine_Learning_Data
import main as glob
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# LESBRIEF Deel 1
# UNSUPERVISED LEARNING
data = Machine_Learning_Data(glob.student_number)

# Haal clustering data op
kmeans_training = data.clustering_training()

# extract de x waarden
X = glob.extract_from_json_as_np_array("x", kmeans_training)

# Slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
x = X[...,0]
y = X[...,1]

# Teken als zwarte punten
for i in range(len(x)):
    plt.plot(x[i], y[i], "k.")
    
plt.show()

# ontdek de clusters mbv kmeans en teken een plot met kleurtjes
colors = ["red", "blue", "green"]
colors.append("yellow")
# colors.append("purple")

kmeans = KMeans(len(colors))
kmeans.fit(X)
clusters = kmeans.cluster_centers_
y_km = kmeans.fit_predict(X)

# Teken de punten in kleur, zodat er clusters te zien zijn
for i in range(len(colors)):
    plt.scatter(X[y_km == i, 0], X[y_km == i, 1], color=colors[i])

# Teken de clusters met centroids
for i in range(len(colors)):
    plt.scatter(X[y_km == i, 0], X[y_km == i, 1], color=colors[i])
    plt.scatter(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1], c="black")

plt.show()