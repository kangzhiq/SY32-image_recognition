#Q5 Arbre de décision

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,6],[2,3],[3,5],[5,4],[0,1],[4,2],[6,0],[7,7]])
Y = np.array([-1,-1,-1,-1,1,1,1,1])

def calculateGini(Y):
    nb_total = len(Y)
    nb_posi = np.sum(np.equal(Y,1))
    p = nb_posi/nb_total
    return 2*p*(1-p)

def seperateData(separateur,dimension,X,Y):
    #point <= separateur
    indice1 = np.less_equal(X[:,dimension],separateur)
    x_less = np.array(X)[np.array(indice1)]
    y_less = np.array(Y)[np.array(indice1)]
    # point > seperateur
    indice2 = np.greater(X[:,dimension],separateur)
    x_greater = np.array(X)[np.array(indice2)]
    y_greater = np.array(Y)[np.array(indice2)]
    return x_less,y_less,x_greater,y_greater

def calculateCART_half(separateur,dimension,X,Y):
    nb_total = len(X)
    x_ls,y_ls,x_gt,y_gt = seperateData(separateur,dimension,X,Y)
    I_ls = calculateGini(np.array(Y)[np.array(np.less_equal(X[:,dimension],separateur))])
    I_gt = calculateGini(np.array(Y)[np.array(np.greater(X[:,dimension],separateur))])    
    nb_ls = np.sum(np.less_equal(X[:,dimension],separateur))
    nb_gt = np.sum(np.greater(X[:,dimension],separateur))
    return nb_ls/nb_total*I_ls + nb_gt/nb_total*I_gt


def findSeparateur(X,Y):
    gini = calculateGini(Y)
    I_max = 0;
    separateur = np.inf
    dimension_sep = -1
    
    for dimension in range(X.shape[1]): 
        potential_seps = np.arange(np.min(X[:,dimension]),np.max(X[:,dimension]),0.5)
        for sep in potential_seps:
            value_temp = gini - calculateCART_half(sep,dimension,X,Y) 
            if value_temp > I_max:
                I_max = value_temp
                separateur = sep
                dimension_sep = dimension 
    return dimension_sep,separateur



def classifier(X,Y):
    separateurs = np.zeros((1,2));
    if(calculateGini(Y) == 0):
        return 
    else:
        dimension,value = findSeparateur(X,Y)
        #separateurs = np.concatenate((separateurs,np.array([[dimension,value]])),axis=0)
        separateurs = np.array([[dimension,value]])
        X1,Y1,X2,Y2 = seperateData(value,dimension,X,Y)
        t1 = classifier(X1,Y1)
        if t1 is not None:
            separateurs = np.concatenate((separateurs,t1),axis=0)
        t2 = classifier(X2,Y2)
        if t2 is not None:
            separateurs = np.concatenate((separateurs,t2),axis=0)
        return separateurs
        
classifier(X,Y)

