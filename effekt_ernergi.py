import numpy as np
from pvlib.location import Location
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from Koordinatsystem import *
from scipy import integrate


def data_load(time_interval, date="2024-04-20"):
    if time_interval == 'yearly'
        tidszone = "Europe/Copenhagen"
        start_dato = "2024-01-01"
        slut_dato = "2025-12-31"
        delta_tid = "h"

    elif time_interval == 'daily'
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
    solpos_angles = np.deg2rad(solpos[["zenith", "azimuth"]].to_numpy())

    # Create a mask for specifik angle
    mask = solpos_angles[:, 0] < np.pi / 2
    solpos_angles = solpos_angles[mask, :]
    # Apply to time aswell
    times = times[mask]

    return solpos_angles, times


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
    theta_panel_full = np.full((m,), theta_p)
    phi_panel_full = np.full((m,), phi_p)

    # Extracting the specfik dimensions of panel plane
    a1, b1 = panel_dim[0]
    a2, b2 = panel_dim[1]
    # initializing an empty array for flux values
    # F = np.empty(m)

    # Returning a list of projections of normalvecktors u_s and u_p
    normal_vector_projection = solar_panel_projection(
        angles_s[:, 0], angles_s[:, 1], theta_panel_full, phi_panel_full
    )
    # for i in range(m):
    # F[i] = sp.integrate((u_sp[i] * S_0), (u, a1, b1), (v, a2, b2))
    # Dividing by 1000 to convert from W/m^2 to kW/m^2
    flux_solar = (normal_vector_projection * S_0 * (b1 - a1) * (b2 - a2)) / 1_000
    return flux_solar


def test(angels, phi_p, theta_p, panel_dim, S_0, int_):
    integral_values = np.empty(len(phi_p))
    for i in range(len(phi_p)):
        F = flux(angels, theta_p[i], phi_p[i], panel_dim, S_0)

        # Only for dayly simulation
        if int_ == 60:
            # Dividing by number of hours in a day to get the KWh/m^2
            integral_values[i] = integrate.simps(F, dx=int_) / (60 * 60 * 24)
        else:
            # Dividing by number of hours in the period to get the KWh/m^2
            integral_values[i] = integrate.simps(F, dx=int_) / (60 * 60 * angels.shape[0])
    max_index = np.argmax(integral_values)
    min_index = np.argmin(integral_values)
    return integral_values, max_index, min_index


# array of phi values including the max and min index
phi_panel = np.linspace(np.pi, np.pi, 91)

# Array of theta values in radians from 0 to 90 degrees
theta_panel = np.radians(np.arange(0, 91, 1))

# Check if the simulation is yearly or hourly
time_interval = 'yearly'

# load data
sun_angles, time = data_load(type=time_interval == 'yearly')

m, n = sun_angles.shape

# Integral period based on yearly or hourly simulation
if time_interval == 'yearly':
    period_seconds = 3600
elif time_interval == 'daily':
    period_seconds = 60

# Defining the panel dimensions
panel_dimensions = ((0, 1), (0, 2))
S_0 = 1_100

flux_total_arr, max_index, min_index = test(sun_angles, phi_panel, theta_panel, panel_dimensions, S_0, int_=period_seconds)


theta_max = theta_panel[max_index]
theta_min = theta_panel[min_index]

# print(F_t)
print(f"Max value: {flux_total_arr[max_index]} at theta = {np.rad2deg(theta_max)} degrees")
print(f"Min value: {flux_total_arr[min_index]} at theta = {np.rad2deg(theta_min)} degrees")
