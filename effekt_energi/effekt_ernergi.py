import numpy as np
from pvlib.location import Location
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from koordinatsystem import *
from scipy import integrate

hourly_consumption = np.array([
    0.3, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4
])

hourly_price = np.array([
    282.27, 259.43, 257.79, 254.89, 252.58, 274.53, 334.94, 394.46,
    432.33, 392.75, 332.34, 285.32, 259.58, 256.75, 213.53, 232.35,
    247.38, 340.75, 394.91, 418.27, 385.68, 338.52, 261.14, 249.76
])

def data_load(
    time_interval, latitude, longitude, tidszone, altitude, date="2024-04-20"
):

    if time_interval == "year":
        start_dato = "2024-01-01"
        slut_dato = "2024-12-31"
        delta_tid = "min"

    elif time_interval == "day":
        start_dato = date
        slut_dato = date
        delta_tid = "min"

    # Definition of Location object. Coordinates and elevation of Amager, Copenhagen (Denmark)
    site = Location(
        latitude, longitude, tidszone, altitude, "Lyngby (DK)"
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

    return solpos_angles, times


def plot_energy_curve(energy_generation: np.ndarray, save: bool):
    # Create the plot
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(energy_generation, color="navy", linewidth=2)  # Customized line
    plt.xlabel(
        "Dage efter jan. 1 ", fontsize=12, fontweight="bold"
    )  # Set the label for the x-axis
    plt.ylabel(
        "Energi (kWh)", fontsize=12, fontweight="bold"
    )  # Set the label for the y-axis
    plt.title(
        "Dagligt energiudbytte for 2024", fontsize=14, fontweight="bold"
    )  # Set the title of the plot
    plt.grid(
        True, which="both", linestyle="--", linewidth=0.5
    )  # Enable grid with custom settings
    plt.tight_layout()  # Adjust layout to not cut off any labels

    # Show the plot
    if save:
        plt.savefig("./img/energy_generation_day_min.png", dpi=300)
    plt.show()


def flux(
    angles_s,
    theta_p: float,
    phi_p: float,
    Area: any,
    S_0: int,
    A_0: float,
    W_p: float,
):
    # u, v = sp.symbols("u v")

    # Defining shape of angles_s array
    m, n = angles_s.shape

    # creating full arrays
    theta_panel_full = np.full((m,), theta_p)
    phi_panel_full = np.full((m,), phi_p)

    # initializing an empty array for flux values

    # Returning a list of projections of normalvecktors u_s and u_p

    normal_vector_projection = solar_panel_projection(
        angles_s[:, 0], angles_s[:, 1], theta_panel_full, phi_panel_full
    )

    # for i in range(m):
    # F[i] = sp.integrate((u_sp[i] * S_0), (u, a1, b1), (v, a2, b2))
    # Convert from W/m^2 to kW
    flux_solar = (normal_vector_projection * (S_0 * A_0) * W_p * Area) / 1_000
    return flux_solar


def test(angles, phi_p, theta_p, panel_area, S_0, A_0, W_p, int_):
    # Initialize arrays for the panel_effekt over time and the integral values
    panel_effekt_vs_time = np.empty((angles.shape[0], theta_p.shape[0]))
    integral_values = np.empty(theta_p.shape[0])

    # Loop over the theta_p (angles of panel) values
    for i in range(len(theta_p)):
        print(f"Calculating for theta = {np.rad2deg(theta_p[i])}")
        # Calculate the flux for each angle
        F = flux(angles, theta_p[i], phi_p, panel_area, S_0, A_0, W_p)

        # Store the flux values for each angle
        panel_effekt_vs_time[:, i] = F

        # Integral over time period for each angle
        if int_ == 60:
            # Dividing by number of hours in a day to get the KWh
            integral_values[i] = integrate.simpson(F, dx=int_) / (60 * 60)
        else:
            # Dividing by number of hours in the period to get the KWh
            integral_values[i] = integrate.simpson(F, dx=int_) / 3600
            # print(integral_values[i])
        print(integral_values[i])
    max_index = np.argmax(integral_values)
    min_index = np.argmin(integral_values)
    return integral_values, panel_effekt_vs_time[:, max_index], max_index, min_index


def energy_per_day(
    angle_values: np.ndarray, theta_p, phi_p, panel_area, S_0, A_0, W_p, int_
):
    # Loop over the theta_p (angles of panel) values
    daily_energy_arr = []

    F = flux(angle_values, theta_p[0], phi_p[0], panel_area, S_0, A_0, W_p)
    F = np.array_split(F, 366)

    for j in range(len(F)):
        daily_energy = integrate.simpson(F[j], dx=int_) / 3600
        daily_energy_arr.append(daily_energy)

    print(
        f"Total energy generation: {np.sum(daily_energy_arr):.3f} for theta angle: {np.rad2deg(theta_p[0]):.3f}"
    )

    return daily_energy_arr


def plot_energy_vs_theta(save: bool):
    flux_total_arr, flux_vs_best_angle, max_index, min_index = test(
        sun_angles, phi_panel, theta_panel, panel_areal, S_0, A_0, W_p, period_seconds
    )
    theta_max = theta_panel[max_index]
    theta_min = theta_panel[min_index]

    # print(F_t)
    print(
        f"Max value: {flux_total_arr[max_index]:.3f} kWh, at theta = {np.rad2deg(theta_max)} degrees"
    )
    print(
        f"Min value: {flux_total_arr[min_index]:.3f} kWh, at theta = {np.rad2deg(theta_min)} degrees"
    )

    # Create the plot
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(
        np.rad2deg(theta_panel), flux_total_arr, color="navy", linewidth=2
    )  # Customized line
    plt.xlabel(
        "Theta Vinkel ", fontsize=12, fontweight="bold"
    )  # Set the label for the x-axis
    plt.ylabel(
        "Energi (kWh)", fontsize=12, fontweight="bold"
    )  # Set the label for the y-axis
    plt.title(
        "Energiudbytte vs Theta Vinkel: 2024", fontsize=14, fontweight="bold"
    )  # Set the title of the plot
    plt.grid(
        True, which="both", linestyle="--", linewidth=0.5
    )  # Enable grid with custom settings
    plt.tight_layout()  # Adjust layout to not cut off any labels

    # Show the plot
    if save:
        plt.savefig("./img/energy_vs_theta_year.png", dpi=300)
    plt.show()


# Check if the simulation is yearly or hourly
time_interval = "year"

# Coordinates for building 101 on DTU Lyngby Campus
latitude = 55.786050  # Breddegrad
longitude = 12.523380  # Længdegrad
altitude = 52  # Meters above sea level. 42 meters is the level + approximately 10 meters for the building height.

# load data
sun_angles, time = data_load(
    time_interval=time_interval,
    latitude=latitude,
    longitude=longitude,
    tidszone="Europe/Copenhagen",
    altitude=altitude,
    date="2024-04-20",
)

# Integral period based on yearly or hourly simulation
if time_interval == "year":
    period_seconds = 60
elif time_interval == "day":
    period_seconds = 60

# array of phi values including the max and min index
phi_panel = np.linspace(np.deg2rad(180), np.deg2rad(180), 1)
# Array of theta values in radians from 0 to 90 degrees
theta_panel = np.radians(np.arange(51, 52, 1))

# Defining the panel dimensions i meters
Længde = 2.278  # længde på solpanel
Bredde = 1.133  # bredde på solpanel
panel_areal = Længde * Bredde  # Areal af solpanelsetup

S_0 = 1_100  # Samlede stråling (irradians)
A_0 = 0.5  # Atmotfæriske forstyrrelser
W_p = 0.211  # Solpanelet effektivitets faktor


# plot_energy_vs_theta(save=True)

energy_generation = energy_per_day(
    sun_angles, theta_panel, phi_panel, panel_areal, S_0, A_0, W_p, period_seconds
)

plot_energy_curve(energy_generation, save=True)

# min_indx = np.argmin(energy_generation)
# max_indx_100 = np.argmax(energy_generation[:175])
# max_indx_250 = np.argmax(energy_generation[175:])
# print(f"Minimum udbytte: {energy_generation[min_indx]} ved dag {min_indx}")
# print(f"Max udbytte 100: {energy_generation[max_indx_100]} ved dag {max_indx_100}")
# print(f"Max udbytte 250: {energy_generation[max_indx_250]} ved dag {max_indx_250}")

# Write the flux values for the best angle to a csv file
# flux_df = pd.DataFrame(flux_vs_best_angle, columns=["Flux"])
# flux_df.to_csv("flux_values.csv")
