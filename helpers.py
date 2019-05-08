from math import sin, cos, sqrt, atan2, radians 
from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import data
import random
from tqdm import tqdm


def distance(city1, city2):
    """Calculates the distance between two longitude latitude pairs in km"""
    # radius of earth in km 
    R = 6373.0

    loc1 = location(city1)
    loc2 = location(city2)

    # convert degrees to radians
    lat1 = radians(loc1[0]) 
    lon1 = radians(loc1[1]) 
    lat2 = radians(loc2[0]) 
    lon2 = radians(loc2[1]) 

    dlon = lon2 - lon1
    dlat = lat2 - lat1 

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2 
    c = 2 * atan2(sqrt(a), sqrt(1 - a)) 

    return R * c


def hometowns():
    """Find the (probable) hometown of each user in the dataset"""
    # init variables
    hometowns = {}

    # find the city of each user
    for city in CITIES:
        for i in USERS[city]:
            hometowns[i["user_id"]] = city
    return hometowns
    

def location(city):
    """Return a tuple with (latitude, longitude) for the city""" 
    # init variables
    lon = []
    lat = []

    # Calculate the average longitude and latitude of the businesses in the city
    for i in BUSINESSES[city]:
        lon.append(i["longitude"])
        lat.append(i["latitude"])
    # make sure a longitude and latitude were provided
    if lon and lat:
        return (sum(lat)/len(lat), sum(lon)/len(lon))
    else:
        return None
        
# def location_user(user_id):
#     """Finds the longitude and latitude of a user based on the locations of previously-reviewed businesses"""
    
#     # init variables
#     long = []
#     lat = []

#     # go over all reviews in all cities
#     for city in tqdm(CITIES):
#         for i in REVIEWS[city]:
#             # if the review was left by the desired user, save the business_id
#             if i["user_id"] == user_id:
#                 business_id = i["business_id"]
#                 # go through all businesses and, if it's the desired business_id, find the exact location
#                 for j in BUSINESSES[city]:
#                     if j["business_id"] == business_id:
#                         long.append(j["longitude"])
#                         lat.append(j["latitude"])
    
#     if len(long) == 0 or len(lat) == 0:
#         return None
#     # return the average latitude and longitude of the businesses
#     return {"latitude" : sum(lat)/len(lat), "longitude" : sum(long)/len(long)}