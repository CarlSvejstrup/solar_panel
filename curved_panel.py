import numpy as np
import sympy as sp
import pvlib
from pvlib.location import Location
import pandas as pd
from Koordinatsystem import angle_to_coords

def flux_of_curve(V_s, V_p, sun_angles):
    # make a array to store the flux
    integrals = np.zeros(V_s[:,0].shape)
    
    # loop through all the sun vectors and sun angles
    for i, (v_s, sun_angle) in enumerate(zip(V_s, sun_angles)):
        # if the sun is under the horizen, the panel has a negative flux, that makes no sense in the real world
        # So if it is under the horizen we do not calcualte it and it stays at 0
        if sun_angle[0] <= np.pi/2:
            # calculate the dot produkt of the vecotr field and the normal vectors of the panel
            dot = np.dot(v_s, V_p)
            
            # make the inner integration of the dot product from 0 to pi/2 using theta
            temp = sp.integrate(dot, (theta_p, 0, sp.pi/2))
            
            # depending on the suns lokation, we will integrate using diffrent limits
            # this is to insure that no part of the panel has a negativ flux, which would not be true to reality
            if sun_angle[1] >= sp.pi:
                integrals[i] = sp.integrate(temp, (phi_p, sun_angle[1]-sp.pi/2, 3*sp.pi/2))
            else:
                integrals[i] = sp.integrate(temp, (phi_p, sp.pi/2, sun_angle[1]+sp.pi/2))
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
    time_interval = "day"

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
    
    V_s = angle_to_coords(sun_angles[:,0], sun_angles[:,1], 1)
    print(flux_of_curve(V_s=V_s, V_p=V_p, sun_angles=sun_angles))