from sklearn.svm import LinearSVC
import numpy as np

xa = np.loadtxt('phishing_train.data.csv',delimiter=' ')
ya = np.loadtxt('phishing_train.label.csv')

clf = LinearSVC()

clf.fit(xa,ya)

taux_erreur = 5000 - np.sum(np.logical_and(clf.predict(xa),ya))
test = 1;
