import sympy as sp
import numpy as np

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
        if (
            np.rad2deg(theta_s) < 90
        ):  # Only calculates the flux if the sun is above the horizon
            flux = max(
                sp.sin(theta_p) * sp.sin(theta_s) * sp.cos(phi_p - phi_s)
                + sp.cos(theta_p) * sp.cos(theta_s),
                0,
            )  # Only returns positive values for the flux
            # print(f"Calculating flux = {round(flux,2)} at theta = {round(np.rad2deg(theta_s),2)}")
        else:
            # print(f"Sun is down")
            flux = 0

        inner[i] = flux
    return inner


theta_sol = np.array([np.pi / 4, np.pi / 2, 0.0, np.pi / 4, np.pi / 4, np.pi / 4])
phi_sol = np.array([np.pi, np.pi / 2, 0.0, np.pi, np.pi, np.pi])
theta_panel = np.array([0.0, np.pi / 2, np.pi, np.pi / 4, np.pi / 4, np.pi / 4])
phi_panel = np.array([np.pi, 0.0, 0.0, 0.0, np.pi, np.pi / 4])


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


"""
Skriv en Python funktion der omregner fra solens position på himlen i et 
 koordinater til zenit og azimuth (i grader eller radianer). Her kan np.arctan2(y, x) og np.rad2deg() være nyttige.
"""


def coords_to_angle(x, y, z):
    return sp.acos(z), sp.atan2(y, x)


if __name__ == "__main__":
    print(solar_panel_projection(theta_sol, phi_sol, theta_panel, phi_panel))
