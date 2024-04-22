import numpy as np
import sympy as sp
import pvlib
from pvlib.location import Location
from scipy import integrate
import pandas as pd
from Koordinatsystem import angle_to_coords

def flux_of_curve(V_s, V_p, sun_angles):
    k=101
    # make a array to store the flux
    integrals = np.zeros(V_s[:,0].shape)
    # loop through all the sun vectors and sun angles
    theta_p_linspace = np.linspace(0, (np.pi/2), k)
    phi_p_linspace = np.linspace(np.pi/2, 3*np.pi/2, k)
    avg_norm_vect = np.zeros((k-1, k-1, 3))
    for i in range(k-1):
        for j in range(k-1):
            avg_norm_vect_x = V_p[0].subs([
                (theta_p, (theta_p_linspace[i]+theta_p_linspace[i+1])/2), 
                (phi_p, (phi_p_linspace[j] + phi_p_linspace[j+1])/2)])
            
            avg_norm_vect_y = V_p[1].subs([
                (theta_p, (theta_p_linspace[i]+theta_p_linspace[i+1])/2), 
                (phi_p, (phi_p_linspace[j] + phi_p_linspace[j+1])/2)])
            
            avg_norm_vect_z = V_p[2].subs([
                (theta_p, (theta_p_linspace[i]+theta_p_linspace[i+1])/2), 
                (phi_p, (phi_p_linspace[j] + phi_p_linspace[j+1])/2)])
            avg_norm_vect[i, j, 0], avg_norm_vect[i, j, 1], avg_norm_vect[i, j, 2] = avg_norm_vect_x, avg_norm_vect_y, avg_norm_vect_z
    for c, (v_s, angle) in enumerate(zip(V_s, sun_angles[:,0])):
        if angle < sp.pi/2:
            # if the sun is under the horizen, the panel has a negative flux, that makes no sense in the real world
            # So if it is under the horizen we do not calcualte it and it stays at 0
            #if sun_angle[0] <= np.pi/2:
                # calculate the dot produkt of the vecotr field and the normal vectors of the panel
            inte = 0

            # make the inner integration of the dot product from 0 to pi/2 using theta

            for i in range(k-1):
                for j in range(k-1):
                    
                    dot = np.dot(avg_norm_vect[i, j], v_s)

                    inte += max(dot * 1/((k-1)**2),0)
            integrals[c] = inte
    return integrals

def data_load(
    time_interval, latitude, longitude, tidszone, altitude, date="2024-07-20"
):

    if time_interval == "year":
        start_dato = "2024-01-01"
        slut_dato = "2024-12-31"
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


if __name__ == "__main__":
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
    phi_p, theta_p = sp.symbols("phi_p, theta_p")
    x_p = sp.cos(phi_p) * sp.sin(theta_p)
    y_p = sp.sin(phi_p) * sp.sin(theta_p)
    z_p = sp.cos(theta_p)
    V_p = np.array([x_p, y_p, z_p])
    # sun_angles = np.array([[sp.pi/4, sp.pi]])
    V_s = angle_to_coords(sun_angles[:,0], sun_angles[:,1], 1)
    F = (flux_of_curve(V_s=V_s, V_p=V_p, sun_angles=sun_angles))
    
    længde = 2.278
    bredde = 1.133
    S_0 = 1_100
    A_0 = 0.5
    WP = 0.211
    F = F*S_0*A_0*WP*længde*bredde/1000
    print(integrate.simpson(F, dx=3600)/3600)