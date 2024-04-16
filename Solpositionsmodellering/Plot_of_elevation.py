from solarposition_models import *
'''Question 3: Plot solens elevationsvinkel og find ud af hvornår på dagen solen
    står højest den 20. april 2024. Forklar hvad det betyder når elevationsvinklen er mindre end 0 grader,
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
elevation_angle_figure = plt.figure(figsize = (4 , 7))
plt.plot(solpos.loc[chosen_date].elevation)
plt.ylabel("Solar elevation angle (degree)")
plt.xlabel("Time (hour)")
plt.title("Solar Elevation Angle in " + site.name +' '+ chosen_date)
plt.xticks(rotation=45)
plt.grid()

# Add the time of sunrise & sunset:
sunset_time = np.argmax(solpos.loc[chosen_date].elevation)
sunrise_time = np.argmin(solpos.loc[chosen_date].elevation)

# Add points at sunrise and sunset
sunrise_x = pd.to_datetime('2024-04-20 05:58:00+02:00' )
sunset_x = pd.to_datetime('2024-04-20 20:21:00+02:00')
plt.plot(sunrise_x, solpos.loc[chosen_date].elevation[sunrise_x], 'ko' , label='Sunrise: 05:58')
plt.plot(sunset_x, solpos.loc[chosen_date].elevation[sunset_x], 'ko', label='Sunset: 20:21')

# Add the legend
plt.legend()

# Add note
note = "Note: Add +2 to the x-axis for Danish time"
plt.text(0.5, 1-0.1, note, ha='center')

# Display the plot
plt.savefig('Elevationangle_{}'.format(chosen_date))
plt.show()

# Explanation of the elevation angle:
# When the elevation angle is less than 0 degrees, the sun is below the horizon (night time).
# When the elevation angle is 0 degrees, the sun is at the horizon (sunrise or sunset).
# When the elevation angle is greater than 0 degrees, the sun is above the horizon (day time).

# Explanation of the zenith angle:
# When the zenith angle is more than 90 degrees, the sun is below the horizon (night time).
# When the zenith angle is 90 degrees, the sun is at the horizon (sunrise or sunset).
# When the zenith angle is less than 90 degrees, the sun is above the horizon (day time).
