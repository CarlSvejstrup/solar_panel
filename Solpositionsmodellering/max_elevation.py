'''Question:
    Lav en Python-funktion som kan beregne solens højeste punkt 
    på himlen (i grader) på en given dato (year-month-day) i en given
    lokation (fx by) angivet ved en breddegrad og længdegrad.
    Hint: Svaret bør ikke afhænge af længdegraden, da solens højeste punkt
    på himlen kun afhænger af breddegraden.'''
import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from timezonefinder import TimezoneFinder


# First we need a function that takes the latitude and longitude and returns the timezone.
def get_timezone(latitude, longitude):
    '''Function that given a latitude and longitude returns the timezone.'''
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
    return timezone_str

# Then we can work on our function:
def max_elevation(date:str , longitude:float , latitude:float , height = 0):
    """Function that calculates the maximum elevation angle of the sun on a given date.
    
    Args:
        date: str : Date in the format "year-month-day"
        longitude: float : Longitude of the location
        latitude: float : Latitude of the location
        height: int : Height of the location

    Returns:
        max_degree: float : The maximum elevation angle of the sun on the given date.
    """

    # Define the time and location to get solar position with pvlib:
    times = pd.date_range(start= date + " 00:00:00",
                          end= date + " 23:59:00", 
                          inclusive="left", 
                          freq="Min", 
                          tz=get_timezone(latitude, longitude))
    
    location = Location(latitude=latitude,
                        longitude=longitude,
                        tz=get_timezone(latitude, longitude),
                        altitude=height)
    
    # Get the solar position:
    solpos = location.get_solarposition(times)

    # Make it as an array:
    elevation_array = solpos.loc[date].apparent_elevation.to_numpy()

    # Find the max elevation angle:
    max_degree = np.max(elevation_array)

    return max_degree