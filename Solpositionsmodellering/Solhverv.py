'''Question 5:Find solens højeste punkt på himlen (i grader) på sommersolhverv på DTU,
    og hvornår på dagen det sker? Hint: Du bliver nødt til at ændre på start og slut dato for solpos-objektet.'''

from coordinates import *
import numpy as np

# For this assignment, we do not what what date to choose. We will find it through checking all days in june.
# Now we need to make a new 'times', since we before import the date 1'st of april till 30'th of april.

# Define the timedates:
start_date = "2024-06-01"
end_date = "2024-06-30"
delta_time = "1min"

# Now for making the times
times = pd.date_range(
    start_date + " 00:00:00", end_date + " 23:59:00", inclusive="left", freq=delta_time, tz=timezone
)
# Get the sunpositions:
solpos = site.get_solarposition(times)

# Make it as an array:
elevation_array = solpos.values[:,2]

# Find the max elevation angle:
max_degree = np.max(elevation_array)

# Find the time of day:
time_index= np.argmax(elevation_array)
time = times[time_index].strftime('%Y-%m-%d %H:%M:%S')


# Display the result:
print("The sun's max angle:" , max_degree)
print("Time:",time)


