import numpy as np
from pvlib.location import Location
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('C:/Users/oscar/OneDrive/School/DTU/Semester_1/Introduktion_til_Intelligente_Systemer/Exam Project/Github/solar_panel/Solpositionsmodellering')
from Koordinatsystem import *
from scipy import integrate
import time as t

# Data from https://www.dst.dk/Site/Dst/SingleFiles/GetArchiveFile.aspx?fi=formid&fo=elforbrug--pdf&ext={2}
hourly_consumption = np.array([
    0.3, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4
])

# Data from https://www.energidataservice.dk/tso-electricity/Elspotprices
# Read the hourly price data from CVS file to np array
df = pd.read_csv('C:/Users/oscar/OneDrive/School/DTU/Semester_1/Introduktion_til_Intelligente_Systemer/Exam Project/Github/solar_panel/Energiforbrug_Optimering/Elspotprices_2023_hourly.csv', header=None)
hourly_price = df.iloc[:, 0].values

# Percent fee for selling electricity back to the power grid
# From https://www.greenmatch.dk/solceller/salg-af-stroem
electricity_selling_fee = 0.05279

def data_load(
    time_interval, latitude, longitude, tidszone, altitude, date="2024-04-20"
):

    if time_interval == "year":
        # Y/M/D
        start_dato = "2023-01-01"
        slut_dato = "2023-12-31"
        delta_tid = "h"

    elif time_interval == "day":
        start_dato = date
        slut_dato = date
        delta_tid = "h"

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

    flux_solar = (normal_vector_projection * (S_0 * A_0) * W_p * Area) / 1_000
    return flux_solar


# def flux_simulation_old(angles, phi_p, theta_p, panel_area, S_0, A_0, W_p, int_):
    # Initialize arrays for the panel_effekt over time and the integral values
    panel_effekt_vs_time = np.empty((angles.shape[0], theta_p.shape[0]))
    integral_values = np.empty(theta_p.shape[0])
    hourly_expense_total = np.empty((angles.shape[0], theta_p.shape[0]))
    expense_over_period = np.empty(theta_p.shape[0])

    # Loop over the theta_p (angles of panel) values
    for i in range(len(theta_p)):
        # Calculate the flux for each angle
        F = flux(angles, theta_p[i], phi_p[i], panel_area, S_0, A_0, W_p)

        # Store the flux values for each angle
        panel_effekt_vs_time[:, i] = F

        # Calculating the actual hourly consumption after solar cell usage
        hourly_expenses_after_solar_cell = np.empty(F.shape[0])
        for l in range(len(F)):
            j = l % 24
            hourly_expense = max((hourly_consumption[j] - F[l]), 0) * hourly_price[j]
            hourly_sales = min((hourly_consumption[j] - F[l]), 0) * hourly_price[j] * electricity_selling_fee * (-1)
            hourly_expenses_after_solar_cell[l] = hourly_expense - hourly_sales

        # Store hourly expense values for each angle
        hourly_expense_total[:, i] = hourly_expenses_after_solar_cell

        # Store expense over the entire period
        expense_over_period[i] = np.sum(hourly_expenses_after_solar_cell)

        # Integral over time period for each angle
        if int_ == 60:
            # Dividing by number of hours in a day to get the KWh
            integral_values[i] = integrate.simpson(F, dx=int_) / (60 * 60)
        else:
            # Dividing by number of hours in the period to get the KWh
            integral_values[i] = integrate.simpson(F, dx=int_) / 3600
        # print(integral_values[i])

    # Storing the maximum values determined by total output and total savings
    max_index_output = np.argmax(integral_values)
    min_index_output = np.argmin(integral_values)
    # "Max" here is the maximized GAIN from the solar panels
    max_index_price = np.argmin(expense_over_period)
    min_index_price = np.argmax(expense_over_period)

    return integral_values, panel_effekt_vs_time, max_index_output, min_index_output, hourly_expense_total, max_index_price, min_index_price


def flux_simulation(angles, phi_p, theta_p, panel_area, S_0, A_0, W_p, int_):
    phi_size = phi_p.shape[0]
    theta_size= theta_p.shape[0]
    total_number_of_hours = angles.shape[0]

    # Initialize arrays for the panel_effekt over time and the integral values
    panel_effekt_vs_time = np.empty((total_number_of_hours, theta_size, phi_size))
    hourly_expense_total = np.empty((total_number_of_hours, theta_size, phi_size))
    integral_values = np.empty((theta_size, phi_size))
    expense_over_period = np.empty((theta_size, phi_size))
    
    progress = 0
    delta_time_progress =  0
    delta_time_progress_limit = 10
    start_time = 0
    end_time = 0
    delta_time = end_time - start_time
    delta_time_avg_data = np.empty((delta_time_progress_limit))
    total_delta_time = 0

    # Loop over both theta and phi values
    for j in range(phi_size):
        for i in range(theta_size):
            start_time = t.time()
            # Calculate the flux for each angle
            F = flux(angles, theta_p[i], phi_p[j], panel_area, S_0, A_0, W_p)

            # Calculating the actual hourly consumption after solar cell usage
            hourly_expenses_after_solar_cell = np.empty(F.shape[0])
            for l in range(len(F)):
                k = l % 24
                hourly_expense = max((hourly_consumption[k] - F[l]), 0) * hourly_price[l]
                hourly_sales = min((hourly_consumption[k] - F[l]), 0) * hourly_price[l] * electricity_selling_fee * (-1)
                hourly_expenses_after_solar_cell[l] = hourly_expense - hourly_sales

            # Store the results in the 3D arrays
            panel_effekt_vs_time[:, i, j] = F
            hourly_expense_total[:, i, j] = hourly_expenses_after_solar_cell

            # Store the integral and expenses in the 2D arrays
            integral_values[i, j] = integrate.simpson(F, dx=int_) / (60 * 60 if int_ == 60 else 3600)
            expense_over_period[i, j] = np.sum(hourly_expenses_after_solar_cell)

            # Progress
            progress += 1
            end_time = t.time()
            total_delta_time += delta_time
            delta_time = end_time - start_time
            delta_time_avg_data[int(delta_time_progress)] = delta_time
            delta_time_avg = np.mean(delta_time_avg_data, axis = 0)
            estimated_remaining_time = (delta_time_avg if delta_time_avg_data[delta_time_progress_limit - 1] != 0 else delta_time) * (phi_size * theta_size - progress)
            delta_time_progress += 1 if delta_time_progress <= (delta_time_progress_limit - 2) else ((delta_time_progress_limit - 1) * (-1))
            print(f"Progress: {round(progress / (phi_size * theta_size) * 100, 2)} % -- Elapsed time: {round(total_delta_time // 3600)} h {round((total_delta_time % 3600) // 60)} min {round(total_delta_time % 60)} s. Estimated remaining time: {round(estimated_remaining_time // 3600)} h {round((estimated_remaining_time % 3600) // 60)} min {round(estimated_remaining_time % 60)} s")

    # Find the indexes of the max and min values across theta and phi
    max_index_output_theta, max_index_output_phi = np.unravel_index(np.argmax(integral_values, axis=None), integral_values.shape)
    min_index_output_theta, min_index_output_phi = np.unravel_index(np.argmin(integral_values, axis=None), integral_values.shape)
    
    max_index_price_theta, max_index_price_phi = np.unravel_index(np.argmin(expense_over_period, axis=None), expense_over_period.shape)
    min_index_price_theta, min_index_price_phi = np.unravel_index(np.argmax(expense_over_period, axis=None), expense_over_period.shape)
    
    # Return the necessary values including the new max and min indexes
    return integral_values, panel_effekt_vs_time, (max_index_output_theta, max_index_output_phi), (min_index_output_theta, min_index_output_phi), hourly_expense_total, (max_index_price_theta, max_index_price_phi), (min_index_price_theta, min_index_price_phi)


# def energy_per_day(angle_values, theta_p, phi_p, panel_area, S_0, A_0, W_p, int_):
    # Loop over the theta_p (angles of panel) values
    daily_energy_arr = []

    F = flux(angle_values, theta_p[0], phi_p[0], panel_area, S_0, A_0, W_p)
    F = np.array_split(F, 365)
    # print(F[0])

    for j in range(len(F)):
        daily_energy = integrate.simpson(F[j], dx=int_) / 3600
        daily_energy_arr.append(daily_energy)

    # plot the daily energy
    plt.plot(daily_energy_arr)
    plt.xlabel("Theta")
    plt.ylabel("Energy (kWh)")
    plt.title("Daily energy for different angles")
    plt.show()

    quit()


###########################
## VARIABLES
###########################

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
    period_seconds = 3_600
elif time_interval == "day":
    period_seconds = 60

# Array of phi values including the max and min index
# 90 degrees = East , 270 degrees = West
phi_panel = np.radians(np.arange(135, 226, 1)) #South-East to South_West
# Array of theta values in radians from 0 to 90 degrees
theta_panel = np.radians(np.arange(0, 91, 1))

# Defining the panel dimensions i meters
# Data from https://www.greenmatch.dk/solceller/6-kw-solcelleanlaeg (approx 36 m2)
Længde = np.sqrt(36)  # længde på solpanel
Bredde = np.sqrt(36)  # bredde på solpanel
panel_areal = Længde * Bredde

S_0 = 1_100  # Samlede stråling (irradians)
A_0 = 0.5  # Atmotfæriske forstyrrelser
W_p = 0.211  # Solpanelet effektivitets faktor

###########################
###########################

# energy_per_day(sun_angles, theta_panel, phi_panel, panel_areal, S_0, A_0, W_p, period_seconds)


# Flux sumulation (New)
flux_total_arr, flux_vs_best_angle_total, max_index_output, min_index_output, hourly_expense_total, max_index_price, min_index_price = flux_simulation(
    sun_angles,
    phi_panel,
    theta_panel,
    panel_areal,
    S_0,
    A_0,
    W_p,
    int_=period_seconds
)

# The best angles for otuput and price
max_output_angles = (theta_panel[max_index_output[0]], phi_panel[max_index_output[1]])
min_output_angles = (theta_panel[min_index_output[0]], phi_panel[min_index_output[1]])
max_price_angles = (theta_panel[max_index_price[0]], phi_panel[max_index_price[1]])
min_price_angles = (theta_panel[min_index_price[0]], phi_panel[min_index_price[1]])

# The best flux and expense values for the best angle determined by OUTPUT
flux_vs_best_angle_output = flux_vs_best_angle_total[:, max_index_output[0], max_index_output[1]]
hourly_expense_output = hourly_expense_total[:, max_index_output[0], max_index_output[1]]
# By PRICE
flux_vs_best_angle_price = flux_vs_best_angle_total[:, max_index_price[0], max_index_price[1]]
hourly_expense_price = hourly_expense_total[:, max_index_price[0], max_index_price[1]]

# Converting the max indexes to string
output_max_str = f"MAX value for OUTPUT: {flux_total_arr[max_index_output[0], max_index_output[1]]:.3f} kWh & Energy Expense per {time_interval} = {round(np.sum(hourly_expense_output) / 100, 2)} DKK ___ theta = {90 - np.rad2deg(max_output_angles[0])}, phi = {np.rad2deg(max_output_angles[1])} degrees"
price_max_str = f"MAX value for PRICE: {flux_total_arr[max_index_price[0], max_index_price[1]]:.3f} kWh & Energy Expense per {time_interval} = {round(np.sum(hourly_expense_price) / 100, 2)} DKK ___ theta = {90 - np.rad2deg(max_price_angles[0])}, phi = {np.rad2deg(max_price_angles[1])} degrees"
output_min_str = f"MIN value for OUTPUT: {flux_total_arr[min_index_output[0], min_index_output[1]]:.3f} kWh ___ theta = {90 - np.rad2deg(min_output_angles[0])}, phi = {np.rad2deg(min_output_angles[1])} degrees"
price_min_str = f"MIN value for PRICE: {flux_total_arr[min_index_price[0], min_index_price[1]]:.3f} kWh ___ theta = {90 - np.rad2deg(min_price_angles[0])}, phi = {np.rad2deg(min_price_angles[1])} degrees"

# Making a consumption per hour reference list for the .csv
hourly_consumption_reference = np.empty(flux_vs_best_angle_output.shape[0])
hourly_expense_reference = np.empty(flux_vs_best_angle_output.shape[0])
for i in range(len(hourly_consumption_reference)):
    hourly_consumption_reference[i] = hourly_consumption[i % 24]
    hourly_expense_reference[i] = hourly_consumption[i % 24] * hourly_price[i]
# Write the hourly expenses with flux data for best angle to csv file
column_ID = list(range(0, len(flux_vs_best_angle_output)))
data = {
    "Hour": column_ID,
    "consumption (kWh)": hourly_consumption_reference,
    "hourly_expense_without_solar_cell": hourly_expense_reference,
    "flux_at_best_angle_OUTPUT": flux_vs_best_angle_output,
    "hourly_expense_OUTPUT": hourly_expense_output,
    "flux_at_best_angle_PRICE": flux_vs_best_angle_price,
    "hourly_expense_PRICE": hourly_expense_price,
}
df = pd.DataFrame(data)

# Adding the index strings to the CSV
extra_info = [output_max_str, price_max_str, output_min_str, price_min_str]
extra_info += [""] * (len(df) - len(extra_info))  # Fill the rest of the column with empty strings
df["Solar Panel Angel Info"] = extra_info

df.to_csv("energy_data.csv", index=False)

# # Plot the flux values for the best angle
# plt.plot(time, flux_vs_best_angle_price)
# plt.xlabel("Time")
# plt.ylabel("Flux (kWh)")
# plt.title("Flux values for the best angle")
# plt.show()


##################
## PRINTING
##################

# print flux determined by output and price
# OUPUT
print(output_max_str)
print(price_max_str)
# PRICE
print(output_min_str)
print(price_min_str)
