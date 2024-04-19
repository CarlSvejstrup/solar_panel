'''Question 4:
    Find tidspunktet for solopgang og solnedgang på DTU den 10. april 2024.
    Sammenlign med “kendte” værdier fx fra DMI. Hint: Hvis I ønsker præcise værdier
    skal I bruge apparent_elevation (apparent sun elevation accounting for atmospheric refraction)
    i stedet for elevation. I behøver ikke tage højde for jordens krumning.'''

from coordinates import *
# Define the date we wish to examine
chosen_date = "2024-04-20"

# Convert the elevation to a numpy array
array_with_elevation = solpos.loc[chosen_date].apparent_elevation.to_numpy()

# Find the index where the elevation is 0: The sun is at the horizon
# Initialize a delta value to minimize
delta = 1
solstice_index_list = np.where(np.logical_and(array_with_elevation < delta , array_with_elevation > -delta))

# Now change delta until you have 2 values, one for sunrise and one for sunset.
delta = 0.05
solstice_index_list = np.where(np.logical_and(array_with_elevation < delta , array_with_elevation > -delta))

# Now we find the exact time of day these indices correspond to.
time_as_array = solpos.loc[chosen_date].index.to_numpy()
sunrise_time = time_as_array[solstice_index_list[0]][0].strftime("%H:%M:%S")
sunset_time = time_as_array[solstice_index_list[0]][1].strftime("%H:%M:%S")

print('Calculated sunrise on', chosen_date, ':', sunrise_time)
print('Calculated Sunset on', chosen_date, ':', sunset_time)

# Compare with www.stjerneskinn.som
sunset_real = '20:27:00'
sunrise_real = '05:51:00'

print('Real sunrise according to "www.sjerneskinn.com" on', chosen_date, ':', sunrise_real)
print('Real Sunset according to "www.sjerneskinn.com" on', chosen_date, ':', sunset_real)

# We can see that the differnce on sunrise is 3 minutes and 1 minute on sunset.
# This is fairly accurate, and the model is working as intended.