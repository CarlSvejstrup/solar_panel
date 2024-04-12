import numpy as np
from pvlib.location import Location
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd


def flux_minute(data):
    theta, phi, x, y, u = sp.symbols("theta phi x y u")
    r = sp.Matrix([x * u, y * u, 0])
    normal = sp.Matrix([])
    inner = V()
    sp.integrate()


if __name__ == "__main__":
    tidszone = "Europe/Copenhagen"
    start_dato = "2024-04-01"
    slut_dato = "2024-04-30"
    delta_tid = "Min"  # "Min", "H",

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

    # Visualize the resulting DataFrame
    # print(solpos.head())

    valgt_dato = "2024-04-20"

    # print(solpos.loc[valgt_dato].zenith)
    # print(solpos.loc[valgt_dato].elevation)
    # print(solpos.loc[valgt_dato].azimuth)

    print(solpos.loc[valgt_dato].zenith)
()
