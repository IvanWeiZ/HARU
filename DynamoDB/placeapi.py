import googlemaps
YOUR_API_KEY = 'AIzaSyAiFpFd85eMtfbvmVNEYuNds5TEF9FjIPI'

from datetime import datetime


gmaps = googlemaps.Client(key='AIzaSyCEJPcVVU9bBh_f757noMSJbz0WsF7joc4')

params = {
    'location': (-23.56, -46.7),
    'radius': 90
}
directions_result = gmaps.places_nearby({'lat':43.76781352, 'lng':-79.46672626},radius=50)
print(directions_result["results"][1])