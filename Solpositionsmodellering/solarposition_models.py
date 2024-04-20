'''Question 2: Plot solens zenit-, azimut- og elevationsvinkel for hele
    dagen den 20. april 2024 som funktion af tiden.'''
from coordinates import *
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

chosen_date = "2024-04-10"

# Plots for solar zenith and solar azimuth angles
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle("Solar Position Estimation in " + site.name +' '+ chosen_date)

# Plot for solar zenith angle
ax1.plot(solpos.loc[chosen_date].zenith)
ax1.set_ylabel("Solar zenith angle (degree)")
ax1.set_xlabel("Time (hour)")
ax1.grid()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

# Plot for solar azimuth angle
ax2.plot(solpos.loc[chosen_date].azimuth)
ax2.set_ylabel("Solar azimuth angle (degree)")
ax2.set_xlabel("Time (hour)")
ax2.grid()
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

# Plot for solar elevation angle
ax3.plot(solpos.loc[chosen_date].elevation)
ax3.set_ylabel("Solar elevation angle (degree)")
ax3.set_xlabel("Time (hour)")
ax3.grid()
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

# Add points at sunrise and sunset
sunrise_x = pd.to_datetime('2024-04-10 06:23:00+02:00' )
sunset_x = pd.to_datetime('2024-04-10 20:01:00+02:00')
ax3.plot(sunrise_x, solpos.loc[chosen_date].elevation[sunrise_x], 'ko' , label='Sunrise: 06:23')
ax3.plot(sunset_x, solpos.loc[chosen_date].elevation[sunset_x], 'ko', label='Sunset: 20:01')
ax2.plot(sunrise_x, solpos.loc[chosen_date].azimuth[sunrise_x], 'ko',label='Sunrise: 06:23')
ax2.plot(sunset_x, solpos.loc[chosen_date].azimuth[sunset_x], 'ko', label='Sunset: 20:01')
ax1.plot(sunrise_x, solpos.loc[chosen_date].zenith[sunrise_x], 'ko',label='Sunrise: 06:23')
ax1.plot(sunset_x, solpos.loc[chosen_date].zenith[sunset_x], 'ko', label='Sunset: 20:01')

# Add the legend
ax1.legend()
ax2.legend()
ax3.legend()

# Add note
note = "Note: Add +2 to the x-axis for Danish time"
fig.text(0.5, 0.025, note, ha='center')

# Display the plot
plt.show()


# Leg
# solpos.loc[chosen_date].elevation['2024-04-10 06:23:00+02:00']

