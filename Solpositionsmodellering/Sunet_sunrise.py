'''Question 4:
    Find tidspunktet for solopgang og solnedgang på DTU den 10. april 2024.
    Sammenlign med “kendte” værdier fx fra DMI. Hint: Hvis I ønsker præcise værdier
    skal I bruge apparent_elevation (apparent sun elevation accounting for atmospheric refraction)
    i stedet for elevation. I behøver ikke tage højde for jordens krumning.'''

from Solpositionsmodellering.coordinates import *
# Define the date we wish to examine
chosen_date = "2024-04-10"

# Convert the elevation to a numpy array
array_with_elevation = solpos.loc[chosen_date].apparent_elevation.to_numpy()

# Find the index where the elevation is 0: The sun is at the horizon
# Initialize a delta value to minimize
delta = 1
solstice_index_list = np.where(np.logical_and(array_with_elevation < delta , array_with_elevation > -delta))

# Now change delta until you have 2 values, one for sunrise and one for sunset.
delta = 0.05
solstice_index_list = np.where(np.logical_and(array_with_elevation < delta , array_with_elevation > -delta))
print('indices for solstice:' , solstice_index_list[0])

# Now we find the exact time of day these indices correspond to.
time_as_array = solpos.loc[chosen_date].index.to_numpy()
sunrise_time = time_as_array[solstice_index_list[0]][0].strftime("%H:%M:%S")
sunset_time = time_as_array[solstice_index_list[0]][1].strftime("%H:%M:%S")

print('Sunrise on', chosen_date, ':', sunrise_time)
print('Sunset on', chosen_date, ':', sunset_time)

# Compare with DMI
sunset_real = '20:06:00'
sunrise_real = '06:16:00'

# 