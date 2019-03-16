import numpy as np
a = np.array([1,2,3,4,5,6])
print(a)
a.size
a.ndim
b = a.reshape((2,3))
b.ndim
print(b)
b+1
b*2
x = np.array([[1],[2],[3]])
print(x)
print('multiplication matricielle')
print(b.dot(x))
c = np.zeros((2,3))
d = np.ones((3,3))
e = np.arange(0,100,10)
c = e.reshape((2,5))
c.sum()
c.max()
c.sum(axis=0)
c.sum(axis=1)
c[0,:]
c[:,1:3]
#boucles et conditions
print('--------------Boucles et conditions---------------')
f = np.random.random(5)
for i in range(f.size):
    if f[i]>0.8:
        print('Grand_nombre : ', f[i])
    elif f[i] < 0.2:
        print('Petit_nombre : ', f[i])
    else:
        print('Au_milieu : ', f[i])
#nombre premier
print('------------------Nombre premier-------------------')
nb_premier = []
n = 2
count = 0
while count < 10:
    for i in range(len(nb_premier)):
        if n%nb_premier[i]==0:
            print(n," n'est pas un nombre premier car ",n,'//',nb_premier[i],'= 0')
            break
    else : 
        print(n,' est un nombre premier')
        count += 1
        nb_premier.append(n)
    n += 1
print('Voici les 10 permiers nombre permier: ')
print(nb_premier)
#Fonctions
print('----------------Functions------------------')
import math
def est_premier(n):
    """Documentation de la fonction : 
        - Input : n un nombre entier naturel
        - Output : True/False"""
    if n < 0 :
        return 'n doit etre un entier naturel, ne peut pas etre un nombre negatif'
    elif math.floor(n) != n:
        return 'non valide car reel'
    elif n == 0|n == 1:
        return False
    else :
        for x in range(2,n):
            if( n % x == 0 ):
                return False
        else:
            return True
print(est_premier.__doc__)

print(est_premier(3))
print(est_premier(20))
print(est_premier(1.33))
print(est_premier(-1.33))
    