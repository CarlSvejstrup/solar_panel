#funktion der kan finder begge nulpunkterne af f.

import numpy as np

def bisect(f,delta):
    c_index = len(f)//2

    if abs((f[0]+f[-1])/2) < delta:
        return c_index
    
    if f[0] * f[-1] <= 0:
        f = f[: c_index]
    else:
        f = f[c_index :]
    return bisect(f,delta)

def find_roots(arr,delta):
    root_arr = []
    split_index = arr.argmin()
    f1 = arr[: split_index]
    f2 = arr[split_index :]
    root_arr.append(bisect(f1,delta=delta))
    root_arr.append(bisect(f2,delta=delta)+split_index)
    return root_arr

t = np.linspace(0, 2 * np.pi, 10000)
f_arr = np.cos(t)

print(find_roots(f_arr,delta=0.001))