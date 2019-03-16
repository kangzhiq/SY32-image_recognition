import numpy as np
import scipy.stats as st

x_positif = st.norm(6,1)
x_negatif = st.norm(4,1)

def calculateError(h):
    return x_positif.cdf(h)*2/3+(1-x_negatif.cdf(h))/3

error_3 = calculateError(3)
error_4 = calculateError(4)