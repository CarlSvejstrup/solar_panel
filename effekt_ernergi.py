import numpy as np
from pvlib.location import Location
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from Koordinatsystem import *
from scipy import integrate


def data_load(type, date="2024-04-20"):
    if type:
        tidszone = "Europe/Copenhagen"
        start_dato = "2024-01-01"
        slut_dato = "2025-12-31"
        delta_tid = "h"

    else:
        tidszone = "Europe/Copenhagen"
        start_dato = date
        slut_dato = date
        delta_tid = "Min"

    # Definition of Location object. Coordinates and elevation of Amager, Copenhagen (Denmark)
    site = Location(
        55.660439, 12.604980, tidszone, 10, "Amager (DK)"
    )  # latitude, longitude, time_zone, altitude, name

    # Definition of a time range of simulation
    times = pd.date_range(
        start_dato + " 00:00:00",
        slut_dato + " 23:59:00",
        inclusive="left",
        freq=delta_tid,
        tz=tidszone,
    )

    # Estimate Solar Position with the 'Location' object
    solpos = site.get_solarposition(times)

    # Convert angles to radians and extract angles into np.array
    angles = np.deg2rad(solpos[["zenith", "azimuth"]].to_numpy())

    # Create a mask for specifik angle
    mask = angles[:, 0] < np.pi / 2
    angles = angles[mask, :]
    # Apply to time aswell
    times = times[mask]

    return angles, times


def flux(
    angles_s,
    theta_p: float,
    phi_p: float,
    panel_dim: tuple,
    S_0: int,
):
    # u, v = sp.symbols("u v")

    # Defining shape of angles_s array
    m, n = angles_s.shape

    # creating full arrays
    theta_p_full = np.full((m,), theta_p)
    phi_p_full = np.full((m,), phi_p)

    # Extracting the specfik dimensions of panel plane
    a1, b1 = panel_dim[0]
    a2, b2 = panel_dim[1]
    # initializing an empty array for flux values
    # F = np.empty(m)

    # Returning a list of projections of normalvecktors u_s and u_p
    u_sp = solar_panel_projection(
        angles_s[:, 0], angles_s[:, 1], theta_p_full, phi_p_full
    )
    # for i in range(m):
    # F[i] = sp.integrate((u_sp[i] * S_0), (u, a1, b1), (v, a2, b2))
    # Dividing by 1000 to convert from W/m^2 to kW/m^2
    F = (u_sp * S_0 * (b1 - a1) * (b2 - a2)) / 1_000
    return F


def test(angels, phi_p, theta_p, panel_dim, S_0, int_):
    int_values = np.empty(len(phi_p))
    for i in range(len(phi_p)):
        F = flux(angels, theta_p[i], phi_p[i], panel_dim, S_0)

        # Only for dayly simulation
        if int_ == 60:
            # Dividing by number of hours in a day to get the KWh/m^2
            int_values[i] = integrate.simps(F, dx=int_) / (60 * 60 * 24)
        else:
            # Dividing by number of hours in the period to get the KWh/m^2
            int_values[i] = integrate.simps(F, dx=int_) / (60 * 60 * angels.shape[0])
    max_index = np.argmax(int_values)
    min_index = np.argmin(int_values)
    return int_values, max_index, min_index


# array of phi values including the max and min index
phi = np.linspace(np.pi, np.pi, 91)

# Array of theta values in radians from 0 to 90 degrees
theta = np.radians(np.arange(0, 91, 1))

# Check if the simulation is yearly or hourly
yearly = True

# load data
angles, time = data_load(type=yearly)

m, n = angles.shape

# Integral period based on yearly or hourly simulation
if yearly:
    period = 3600
else:
    period = 60

# Defining the panel dimensions
panel_dimensions = ((0, 4), (0, 5))
S_0 = 1_100

F_t, max_index, min_index = test(angles, phi, theta, panel_dimensions, S_0, int_=period)


theta_max = theta[max_index]
theta_min = theta[min_index]

# print(F_t)
print(f"Max value: {F_t[max_index]} at theta = {np.rad2deg(theta_max)} degrees")
print(f"Min value: {F_t[min_index]} at theta = {np.rad2deg(theta_min)} degrees")
