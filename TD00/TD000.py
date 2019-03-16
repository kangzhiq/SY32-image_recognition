import numpy as np
import matplotlib.pyplot as plt
data_train = np.loadtxt('SY32_P19_TD01_data.csv')

def calculateErreurEmpirique(data,h):
    erreur = 0
    for i in range(data.shape[0]):
        if(classifieur(data[i,0],h) != data[i,1]):
            erreur += 1
    return erreur/len(data)
def classifieur(value,h):
    if(value <= h):
        return -1
    else:
        return 1    

erreur3 = calculateErreurEmpirique(data_train,3)
erreur4 = calculateErreurEmpirique(data_train,4) 

def plotErreur(data):
    plt.figure()
    x = np.arange(0,10,0.1)
    y = calculateErreurEmpirique(data,x)
    plt.plot(x,y)
    plt.xlabel('h')
    plt.ylabel('Erreur')
    plt.show()
plotErreur(data_train)
        
 
