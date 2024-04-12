import sympy as sp
def solar_elevation_angle(theta):
    return 90-theta

"""Antag at solen har en fast afstand til jorden. Find en rimelig værdi for.
Angiv et (matematisk) udtryk for hvordan solens-koordinat kan udregnes ud fra,
og, hvor og er hhv. zenit og azimut-vinklen for solens placering."""
r_s = 100_000_000_000 # afstand fra jorden til solen i meter
theta_s, phi_s = sp.symbols("theta_s, phi_s")
x = sp.cos(phi_s) * sp.sin(theta_s) * r_s
y = sp.sin(phi_s) * sp.sin(theta_s) * r_s
z = sp.cos(theta_s) * r_s
coordinates_s = (x, y, z)
"""Angiv et (matematisk) udtryk for og for ud fra zenit- og azimut-vinklerne.
I bør simplificere udtrykket så det indeholder og kun 5 trigonometriske funktioner.
Vis at. Forklar man egne ord hvad det betyder når."""
theta_p, phi_p = sp.symbols("theta_s, phi_s")
x = sp.cos(phi_p) * sp.sin(theta_p)
y = sp.sin(phi_p) * sp.sin(theta_p)
z = sp.cos(theta_p)
coordinates_s = (x, y, z)