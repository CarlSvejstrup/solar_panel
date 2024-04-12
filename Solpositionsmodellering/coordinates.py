# Import relevant packages
import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

'''Question 1: Vælg placering/lokation for jeres solpanel, fx DTU.
Ret overstående GPS koordinater (målt i DecimalDegrees), 
højde og navn så det passer med den valgte lokation.'''

# Placering: Building 101, DTU Lyngby Campus

# Coordinates for building 101 on DTU Lyngby Campus
latitude = 55.786050 # Breddegrad
longitude = 12.523380 # Længdegrad
altitude = 52 # Meters above sea level. 42 meters is the level + approximately 10 meters for the building height.

# Timezone for Copenhagen, Denmark
timezone = "Europe/Copenhagen"
start_date = "2024-04-01"
slut_date = "2024-04-30"
delta_time = "Min"  # "Min", "H",

# Definition of Location object. Coordinates and elevation of Amager, Copenhagen (Denmark)
site = Location(
    latitude=latitude, longitude=longitude,tz=timezone, altitude=altitude, name= "Lyngby (DK)"
)

# Definition of a time range of simulation
times = pd.date_range(
    start_date + " 00:00:00", slut_date + " 23:59:00", inclusive="left", freq=delta_time, tz=timezone
)

# Estimate Solar Position with the 'Location' object
solpos = site.get_solarposition(times)

# Visualize the resulting DataFrame
solpos.head()
