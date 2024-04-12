from solarposition_models import *
'''Question 3: Plot solens elevationsvinkel og find ud af hvornår på dagen solen
    står højest den 10. april 2024. Forklar hvad det betyder når elevationsvinklen er mindre end 0 grader,
    og zenit er mindre end 90 grader.'''

# We start by calculating the max height of the sun. The max height of the sun, is
# when the elevation angle is at its highest.
elevation_as_np_array = solpos.loc[chosen_date].elevation.to_numpy()
max_index = np.argmax(elevation_as_np_array)

# Now we find the time of the day when the sun is at its highest.
time_as_array = solpos.loc[chosen_date].index.to_numpy()
time_with_max_sun = time_as_array[max_index].strftime("%H:%M:%S")
print('Time of day with max sun on' , chosen_date , ':' , time_with_max_sun)

# Plot for elevation angle
elevation_angle_figure = plt.figure(figsize = (20/3 , 8))
plt.plot(solpos.loc[chosen_date].elevation)
plt.ylabel("Solar elevation angle (degree)")
plt.xlabel("Time (hour)")
plt.title("Solar Elevation Angle in " + site.name +' '+ chosen_date)
plt.xticks(rotation=45)
plt.grid()

# Add the max sun point
plt.plot(time_as_array[max_index], elevation_as_np_array[max_index], 'ko', label='Max Sun: ' + time_with_max_sun)
plt.legend()

# Display the plot
plt.show()

# Explanation of the elevation angle:
# When the elevation angle is less than 0 degrees, the sun is below the horizon (night time).
# When the elevation angle is 0 degrees, the sun is at the horizon (sunrise or sunset).
# When the elevation angle is greater than 0 degrees, the sun is above the horizon (day time).

# Explanation of the zenith angle:
# When the zenith angle is more than 90 degrees, the sun is below the horizon (night time).
# When the zenith angle is 90 degrees, the sun is at the horizon (sunrise or sunset).
# When the zenith angle is less than 90 degrees, the sun is above the horizon (day time).
