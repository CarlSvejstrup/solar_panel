import sympy as sp
import numpy as np
import pvlib
from pvlib.location import Location
import pandas as pd

# zenit = theta
# azimuth =phi


def solar_elevation_angle(theta):
    return 90 - theta


"""Antag at solen har en fast afstand til jorden. Find en rimelig værdi for.
Angiv et (matematisk) udtryk for hvordan solens-koordinat kan udregnes ud fra,
og, hvor og er hhv. zenit og azimut-vinklen for solens placering."""
r_s = 100_000_000_000  # afstand fra jorden til solen i meter
theta_s, phi_s = sp.symbols("theta_s, phi_s")
x_s = sp.cos(phi_s) * sp.sin(theta_s)
y_s = sp.sin(phi_s) * sp.sin(theta_s)
z_s = sp.cos(theta_s)
coordinates_s = (x_s * r_s, y_s * r_s, z_s * r_s)
"""Angiv et (matematisk) udtryk u_p for og for <u_p,u_s> ud fra zenit- og azimut-vinklerne.
I bør simplificere udtrykket så det indeholder og kun 5 trigonometriske funktioner.
Vis at. Forklar man egne ord hvad det betyder når."""
theta_p, phi_p = sp.symbols("theta_s, phi_s")
theta_p, phi_p = 30, 180
x_p = sp.cos(phi_p) * sp.sin(theta_p)
y_p = sp.sin(phi_p) * sp.sin(theta_p)
z_p = sp.cos(theta_p)
u_p = sp.Matrix([x_p, y_p, z_p])

"""x = -sp.cos(phi_s) * sp.sin(theta_s)
y = -sp.sin(phi_s) * sp.sin(theta_s)
z = -sp.cos(theta_s)
u_s = sp.Matrix(x,y,z)
u_s = sp.Matrix(x,y,z) * 1/sp.sqrt(u_s.T @ u_s) # normalizere vectoren
"""

Inner_u_p_and_u_s = -sp.sin(theta_p) * sp.sin(theta_s) * sp.cos(phi_p - phi_s) - sp.cos(
    theta_p
) * sp.cos(theta_s)
""""
u_p og u_s begge har længden 1, når du finder inder produktet finder man abselut værdien længden af den ene projekteret ind på den anden
i anden. 
Hvis de to vectore er ens vil det indre produkt være 1^2 hvilet er 1. Hvis de to vectore peger modsat vej vil det være -1,
da den abselutte længde er den samme, men den peger den anden vej


Hvis <u_p, u_s> er 0 betyder det de to vecore er vinkelrette på hindanden og der er derfor ikke noget sollys der rammer solcellen
"""


"""
Skriv en Python-funktion def solar_panel_projection(theta_sol, phi_sol, theta_panel, phi_panel) der returnerer 
<n_s, n_p> når det er positivt og ellers returnerer nul.
"""


def solar_panel_projection_single(theta_sol, phi_sol, theta_panel, phi_panel):
    inner = sp.sin(theta_panel) * sp.sin(theta_sol) * sp.cos(
        phi_panel - phi_sol
    ) + sp.cos(theta_panel) * sp.cos(theta_sol)
    if inner > 0:
        return inner
    return inner


"""ig igen på jeres Python-funktion def solar_panel_projection(theta_sol, phi_sol, theta_panel, phi_panel).
Skriv den om så den virker på NumPy-arrays af zenit- og azimut-vinkler.
Du kan teste den på følgende de tre situationer,
hvor projektionen bør give 0.707107, 0.0 og 0.0 (eller rettere,
med numeriske fejl, bør det give array([7.07106781e-01, 6.12323400e-17, 0.0])).
Forklar solpanelets orientering og solens placering i de tre situationer."""


def solar_panel_projection(theta_sol, phi_sol, theta_panel, phi_panel):
    inner = np.zeros(theta_sol.size)
    for i, (theta_s, phi_s, theta_p, phi_p) in enumerate(
        zip(theta_sol, phi_sol, theta_panel, phi_panel)
    ):
        temp = sp.sin(theta_p) * sp.sin(theta_s) * sp.cos(phi_p - phi_s) + sp.cos(
            theta_p
        ) * sp.cos(theta_s)
        if temp > 0:
            inner[i] = temp
    return inner



# to sidste opgave i Solpositionsmodellering ved Pvlib

"""
Skriv en Python-funktion (til brug med NumPy arrays) der omregner fra solens zenit og azimuth til solens position angivet i 
-koordinaten. Husk om I regner i radianer eller grader. Her kan np.deg2rad()-funktionen være nyttig.
Det er fint at bruge en cirka værdi for men man kan finde en mere korrekt værdi ved:
pvlib.solarposition.nrel_earthsun_distance(times) * 149597870700,
hvor 149597870700 er antal meter på en astronomisk enhed AU.
"""


def angle_to_coords(theta_s, phi_s, r_s=100_000_000_000):
    x_s = sp.cos(phi_s) * sp.sin(theta_s)
    y_s = sp.sin(phi_s) * sp.sin(theta_s)
    z_s = sp.cos(theta_s)
    return x_s * r_s, y_s * r_s, z_s * r_s

def angle_to_coords(theta_s, phi_s, r_s=100_000_000_000):
    x_s = np.cos(phi_s) * np.sin(theta_s)
    y_s = np.sin(phi_s) * np.sin(theta_s)
    z_s = np.cos(theta_s)
    return np.array([x_s,y_s,z_s]).T

"""
Skriv en Python funktion der omregner fra solens position på himlen i et 
 koordinater til zenit og azimuth (i grader eller radianer). Her kan np.arctan2(y, x) og np.rad2deg() være nyttige.
"""


def coords_to_angle(x, y, z):
    return sp.acos(z), sp.atan2(y, x)

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