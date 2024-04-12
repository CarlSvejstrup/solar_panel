import numpy as np
from pvlib.location import Location
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

theta_s, phi_s, theta_p, phi_p, x, y, u = sp.symbols(
    "theta_s phi_s theta_p phi_p x y u"
)

# Max iridians
S_0 = 1_100


def unit_postion(phi, theta):
    x = sp.cos(phi_s) * sp.sin(theta_s)
    y = sp.sin(phi_s) * sp.sin(theta_s)
    z = sp.cos(theta_s)

    return sp.Matrix([x, y, z])


# This is the normal unit vector for the sun, based on its current position
u_s = unit_postion(phi_s, theta_s)
# This is normal unit vektor for the panel, based on its position
u_s = unit_postion(phi_p, theta_p)

# The vektorfield for the sun
V_s = u_s * S_0

print(V_s)
# Parameterfremstilling

# Vektorfeltet for V må være givet som S_0 * u_s, hvor u_s er normalvektoren
# Vektorfeltet må være konstant, er bestemt på baggrund af solens normalvektor ned på origo u_s

# r = sp.Matrix([x*u])

# def flux_minute(data):
#     r = sp.Matrix([x * u, y * u, 0])
#     normal = sp.Matrix([])
#     inner = V()
#     sp.integrate()


# if __name__ == "__main__":
#     tidszone = "Europe/Copenhagen"
#     start_dato = "2024-04-01"
#     slut_dato = "2024-04-30"
#     delta_tid = "Min"  # "Min", "H",

#     # Definition of Location object. Coordinates and elevation of Amager, Copenhagen (Denmark)
#     site = Location(
#         55.660439, 12.604980, tidszone, 10, "Amager (DK)"
#     )  # latitude, longitude, time_zone, altitude, name

#     # Definition of a time range of simulation
#     times = pd.date_range(
#         start_dato + " 00:00:00",
#         slut_dato + " 23:59:00",
#         inclusive="left",
#         freq=delta_tid,
#         tz=tidszone,
#     )

#     # Estimate Solar Position with the 'Location' object
#     solpos = site.get_solarposition(times)

#     # Visualize the resulting DataFrame
#     # print(solpos.head())

#     valgt_dato = "2024-04-20"

#     # print(solpos.loc[valgt_dato].zenith)
#     # print(solpos.loc[valgt_dato].elevation)
#     # print(solpos.loc[valgt_dato].azimuth)

#     print(solpos.loc[valgt_dato].zenith)
# ()
