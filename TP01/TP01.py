from sklearn.svm import LinearSVC
import numpy as np

xa = np.loadtxt('phishing_train.data.csv',delimiter=' ')
ya = np.loadtxt('phishing_train.label.csv')

clf = LinearSVC()

clf.fit(xa,ya)

clf.predict(xa)

#Q4
taux_erreur = np.not_equal(clf.predict(xa),ya).sum()/len(xa)

#Q5

#Q6
vc_data = np.concatenate((xa, ya.reshape(5000, 1)),axis=1)


def perform_vc(n, data, c=1):
    clf = LinearSVC(C=c)
    global_err_vc = 0
    for i in range(0,n):
        vc_test = np.zeros((1000,31))
        vc_train = np.zeros((4000,31))
        indice_test = 0
        indice_train = 0
        for j in range(0,5000):
            if j%5 == i:
                vc_test[indice_test,:] = data[j]
                indice_test += 1
            else:
                vc_train[indice_train, :] = data[j]
                indice_train += 1
        clf.fit(vc_train[:,0:29], vc_train[:,30])
        vc_err = np.not_equal(clf.predict(vc_test[:, 0:29]), vc_test[:,30]).sum()/len(vc_test)
        global_err_vc += vc_err
    return global_err_vc/n


for i in [10**-3, 10**-2, 10**-1, 1, 10, 100, 1000]:
    taux_err_vc = perform_vc(5, vc_data, i)   
    print("Avec i = ", i, " taux d'erreur est: ", taux_err_vc)


