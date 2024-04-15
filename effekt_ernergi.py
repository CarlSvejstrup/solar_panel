import numpy as np
from pvlib.location import Location
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from Koordinatsystem import *
from scipy import integrate


def data_load(time_interval, date="2024-04-20"):
    if time_interval == "year":
        tidszone = "Europe/Copenhagen"
        start_dato = "2024-01-01"
        slut_dato = "2024-01-20"
        delta_tid = "h"

    elif time_interval == "day":
        tidszone = "Europe/Copenhagen"
        start_dato = date
        slut_dato = date
        delta_tid = "min"

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
    solpos_angles_test = solpos_angles[mask, :]
    print(solpos_angles_test)
    solpos = np.where(solpos_angles[:, 0] < np.pi / 2, solpos_angles[:, 0], 0)
    print(solpos)

    # Apply to time aswell
    # times = times[mask]

    return solpos_angles, times


def flux(angles_s, theta_p: float, phi_p: float, panel_dim: tuple, S_0: int, A_0):
    # u, v = sp.symbols("u v")

    # Defining shape of angles_s array
    m, n = angles_s.shape

    # creating full arrays
    theta_panel_full = np.full((m,), theta_p)
    phi_panel_full = np.full((m,), phi_p)

    # Extracting the specfik dimensions of panel plane
    Length, Width = panel_dim
    # initializing an empty array for flux values
    # F = np.empty(m)

    # Returning a list of projections of normalvecktors u_s and u_p

    normal_vector_projection = solar_panel_projection(
        angles_s[:, 0], angles_s[:, 1], theta_panel_full, phi_panel_full
    )
    # for i in range(m):
    # F[i] = sp.integrate((u_sp[i] * S_0), (u, a1, b1), (v, a2, b2))
    # Convert from W/m^2 to kW
    flux_solar = (normal_vector_projection * (S_0 * A_0) * (Length * Width)) / 1_000
    print(flux_solar.shape[0])
    return flux_solar


def test(angles, phi_p, theta_p, panel_dim, S_0, A_0, int_):
    integral_values = np.empty(len(theta_p))
    for i in range(len(theta_p)):
        F = flux(angles, theta_p[i], phi_p[i], panel_dim, S_0, A_0)
        # Only for dayly simulation
        if int_ == 60:
            # Dividing by number of hours in a day to get the KWh
            integral_values[i] = integrate.simpson(F, dx=int_)
        else:
            # Dividing by number of hours in the period to get the KWh
            integral_values[i] = (
                integrate.simpson(F, dx=int_) / 3600
            )  #  / (angles.shape[0] * 60 * 60)
        # print(integral_values[i])

    max_index = np.argmax(integral_values)
    min_index = np.argmin(integral_values)
    return integral_values, max_index, min_index


# array of phi values including the max and min index
phi_panel = np.linspace(np.deg2rad(180), np.deg2rad(180), 91)

# Array of theta values in radians from 0 to 90 degrees
theta_panel = np.radians(np.arange(0, 91, 1))

# Check if the simulation is yearly or hourly
time_interval = "year"

# load data
sun_angles, time = data_load(time_interval=time_interval)

m, n = sun_angles.shape

# Integral period based on yearly or hourly simulation
if time_interval == "year":
    period_seconds = 3600
elif time_interval == "day":
    period_seconds = 60

# Defining the panel dimensions i meters
panel_dimensions = (1, 2)

S_0 = 1_000
A_0 = 0.5

# flux_total_arr, max_index, min_index = test(
#     sun_angles, phi_panel, theta_panel, panel_dimensions, S_0, A_0, int_=period_seconds
# )


# theta_max = theta_panel[max_index]
# theta_min = theta_panel[min_index]

# # print(F_t)
# print(
#     f"Max value: {flux_total_arr[max_index]:.3f} kWh, at theta = {np.rad2deg(theta_max)} degrees"
# )
# print(
#     f"Min value: {flux_total_arr[min_index]:.3f} kWh, at theta = {np.rad2deg(theta_min)} degrees"
# )
