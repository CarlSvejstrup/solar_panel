'''Question 2: Plot solens zenit-, azimut- og elevationsvinkel for hele
    dagen den 20. april 2024 som funktion af tiden.'''

from coordinates import *
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

chosen_date = "2024-04-20"

# Use pandas funciton 'Timedelta' to shift the time +2 hours, to fit danish time
adjusted_solpos = solpos.loc[chosen_date].copy()  # Make a copy of the slice to avoid SettingWithCopyWarning
adjusted_solpos.index += pd.Timedelta(hours=2)  # Add 2 hours

# Plots for solar zenith and solar azimuth angles
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Solar Position Estimation in " + site.name +' '+ chosen_date)

# Plot for solar zenith angle
ax1.plot(adjusted_solpos.loc[chosen_date].zenith , )
ax1.set_ylabel("Solar zenith angle (degree)")
ax1.set_xlabel("Time (hour)")
ax1.grid()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

# Plot for solar azimuth angle
ax2.plot(adjusted_solpos.loc[chosen_date].azimuth)
ax2.set_ylabel("Solar azimuth angle (degree)")
ax2.set_xlabel("Time (hour)")
ax2.grid()
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

# Plot for solar elevation angle
ax3.plot(adjusted_solpos.loc[chosen_date].elevation)
ax3.set_ylabel("Solar elevation angle (degree)")
ax3.set_xlabel("Time (hour)")
ax3.grid()
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

# Add points at sunrise and sunset
sunrise_x = pd.to_datetime('2024-04-20 05:58:00+00:00' )
sunset_x = pd.to_datetime('2024-04-20 20:21:00+00:00')

# Finding the sunrise y-value in the plot:
sunrise_y = adjusted_solpos.loc[chosen_date].elevation[sunrise_x]
sunset_y = adjusted_solpos.loc[chosen_date].elevation[sunset_x]

# Plot the points
ax3.plot(sunrise_x, sunrise_y, 'ko' , label='Sunrise: 05:58')
ax3.plot(sunset_x, sunset_y, 'ko', label='Sunset: 20:21')
ax2.plot(sunrise_x, adjusted_solpos.loc[chosen_date].azimuth[sunrise_x], 'ko',label='Sunrise: 05:58')
ax2.plot(sunset_x, adjusted_solpos.loc[chosen_date].azimuth[sunset_x], 'ko', label='Sunset: 20:21')
ax1.plot(sunrise_x, adjusted_solpos.loc[chosen_date].zenith[sunrise_x], 'ko',label='Sunrise: 05:58')
ax1.plot(sunset_x, adjusted_solpos.loc[chosen_date].zenith[sunset_x], 'ko', label='Sunset: 20:21')

# Add the legend
ax1.legend()
ax2.legend()
ax3.legend()

# Add note
# note = "Note: Add +2 to the x-axis for Danish time"
# fig.text(0.5, 1-0.1, note, ha='center')

# Display the plot
plt.savefig('All_3_angles_{}'.format(chosen_date))
plt.show()


# Leg
# solpos.loc[chosen_date].elevation['2024-04-20 20:21:00+02:00']

