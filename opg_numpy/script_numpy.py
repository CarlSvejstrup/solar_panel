#funktion der kan finder begge nulpunkterne af f.

import numpy as np

def bisect(t_min,t_max,f,delta):
    t_mid = (t_min + t_max)/2

    if (t_max - t_min)/2 < delta:
        return t_mid
    
    if f[t > t_min][0] * f[t > t_mid][0] <= 0:
        return bisect(t_min,t_mid,f,delta)
    else:
        return bisect(t_mid,t_max,f,delta)

def find_arr_eq_zero(arr,t_min,t_max,delta):
    root_arr = []
    tf_min = t[arr.argmin()]
    root_arr.append(bisect(t_min,tf_min,arr,delta=delta))
    root_arr.append(bisect(tf_min,t_max,arr,delta=delta))
    return root_arr

t = np.linspace(0, 2 * np.pi, 10000)
f_arr = np.cos(t)
t_min = np.min(t)
t_max = np.max(t)

find_arr_eq_zero(f_arr,t_min,t_max,delta=0.000001)