from machinelearningdata import Machine_Learning_Data
import main as glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# LESBRIEF Deel 2.1
# SUPERVISED LEARNING
# Logestic regression

# Data ophalen
data = Machine_Learning_Data(glob.student_number)

# Classificatie trainingsdata
classification_training = data.classification_training()

# Extract de data x = array met waarden, y = classificatie 0 of 1
X = glob.extract_from_json_as_np_array("x", classification_training)

# Dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = glob.extract_from_json_as_np_array("y", classification_training)

# Leer de classificatie logistic regression
clf = LogisticRegression()
clf.fit(X, Y)

# Plot data
for i in range(len(X)):
    if(Y[i] == 0):
        plt.plot(X[i][0], X[i][1], "r.")
    else:
        plt.plot(X[i][0], X[i][1], "g.")

plt.show()

# Voorspel Y-waarde door te doen alsof je alleen de X hebt
predicted_y = clf.predict(X)

# Plot het resultaat van classifier logistic regression
for i in range(len(X)):
    if(predicted_y[i] == 0):
        plt.plot(X[i][0], X[i][1], "r.")
    else:
        plt.plot(X[i][0], X[i][1], "g.")

plt.show()

# Vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt
print(accuracy_score(Y, predicted_y, normalize=False))

# Classificatie testdata
classification_test = data.classification_test()

# Testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = glob.extract_from_json_as_np_array("x", classification_test)

# Voorspel Y-waarde terwijl je alleen de X hebt
Z = clf.predict(X_test)

for i in range(len(X_test)):
    if(Z[i] == 0):
        plt.plot(X_test[i][0], X_test[i][1], "r.")
    else:
        plt.plot(X_test[i][0], X_test[i][1], "g.")

plt.show()

# Stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist())

plt.show()

print("Classificatie accuratie (test): " + str(classification_test))