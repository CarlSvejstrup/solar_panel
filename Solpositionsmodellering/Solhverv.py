'''Question 5:
    Skriv en Python-funktion (til brug med NumPy arrays) der omregner fra solens zenit og azimuth
    til solens position angivet i xyz-koordinate4. Husk om I regner i radianer eller grader. Her kan np.deg2rad()-funktionen være nyttig.
    Det er fint at bruge en cirka værdi for r_s men man kan finde en mere korrekt værdi ved:
    pvlib.solarposition.nrel_earthsun_distance(times) * 149597870700
    , hvor 149597870700 er antal meter på en astronomisk enhed AU.'''

from coordinates import *
import numpy as np

# First we chose the date of summer soltice according to "Illustreret Videnskab"
chosen_date = "2023-06-20"

# Now we need to make a new 'times', since we before import the date 1'st of april till 30'th of april.

# Define the timedates:
start_date = "2023-06-20"
end_date = "2023-06-21"
delta_time = "1min"

# Now for making the times
times = pd.date_range(
    start_date + " 00:00:00", slut_date + " 23:59:00", inclusive="left", freq=delta_time, tz=timezone
)
# Get the sunpositions:
solpos = site.get_solarposition(times)

# Make it as an array:
elevation_array = solpos.loc[chosen_date].apparent_elevation.to_numpy()

# Find the max elevation angle:
max_degree = np.max(elevation_array)

# Find the time of day:
time_of_day_index= np.argmax(elevation_array)
time_of_day = times[time_of_day_index].strftime("%H:%M:%S")


# Display the result:
print("The sun's max angle:" , max_degree)
print("Time of day:",time_of_day)


