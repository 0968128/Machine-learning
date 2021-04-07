from machinelearningdata import Machine_Learning_Data
import main as glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# LESBRIEF Deel 2
# SUPERVISED LEARNING
data = Machine_Learning_Data(glob.student_number)

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X = glob.extract_from_json_as_np_array("x", classification_training)

# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = glob.extract_from_json_as_np_array("y", classification_training)

# Plot data
for i in range(len(X)):
    plt.plot(X, Y, "k.")

plt.show()

# TODO: leer de classificaties
clf = GaussianNB()
clf.fit(X, Y)

# TODO: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#       bijvoordeeld Y_predict
predicted_y = clf.predict(X)

# TODO: vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt
correct_guesses = []
for i in range(len(predicted_y)):
    if(predicted_y[i] == Y[i]):
        correct_guesses.append(predicted_y[i])

print("Classificatie accuratie (echte Y): " + str(len(correct_guesses) / len(Y)))

# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = glob.extract_from_json_as_np_array("x", classification_test)

# TODO: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
#       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
#       geen echte Y-waarden gekregen hebt.
#       onderstaande code stuurt je voorspelling naar de server, die je dan
#       vertelt hoeveel er goed waren.

Z = clf.predict(X_test)

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie (test): " + str(classification_test))