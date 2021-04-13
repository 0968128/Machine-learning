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

# Extract de x waarden
X = glob.extract_from_json_as_np_array("x", kmeans_training)

# Slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
x = X[...,0]
y = X[...,1]

# Teken als zwarte punten
for i in range(len(x)):
    plt.plot(x[i], y[i], "k.")

plt.xlim(-40, 80)
plt.ylim(-20, 140)
plt.show()

# ontdek de clusters mbv kmeans en teken een plot met kleurtjes
cluster_amount_guess = 4
colors = ["red", "blue", "green", "yellow", "orange", "pink", "purple"]

kmeans = KMeans(cluster_amount_guess)
kmeans.fit(X)
clusters = kmeans.cluster_centers_
y_km = kmeans.fit_predict(X)

# Teken de punten in kleur, zodat het geschatte aantal clusters te zien is
for i in range(cluster_amount_guess):
    plt.scatter(X[y_km == i, 0], X[y_km == i, 1], color=colors[i])

plt.xlim(-40, 80)
plt.ylim(-20, 140)
plt.show()

# Teken de centroids voor mijn gok
for i in range(cluster_amount_guess):
    plt.scatter(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1], color=colors[i])

plt.xlim(-40, 80)
plt.ylim(-20, 140)
plt.show()