#Funktion der kan finder begge nulpunkterne af f.

import numpy as np

def bisect(t_min,t_max,f,delta):
    t_mid = (t_min + t_max)/2

    if (t_min - t_max)/2 < delta:
        return t_mid
    
    if f[t_min] * f[t_mid] <= 0:
        return bisect(t_min,t_mid,delta)
    else:
        return bisect(t_mid,t_max,delta)

def find_f_eq_zero(f,t,delta):
    zero_arr = []
    t_min = np.min(t)
    t_max = np.max(t)
    tf_min = t[f.argmin()]
    zero_arr.append(bisect(t_min,tf_min,f,delta))
    zero_arr.append(bisect(tf_min,t_max,f,delta))
    return zero_arr

t = np.linspace(0, 2 * np.pi, 1000)
f = np.cos(t)

print(find_f_eq_zero(f,t,delta=0.1))
