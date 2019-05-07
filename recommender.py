from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import random

def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """
    if not city:
        city = random.choice(CITIES)
    return random.sample(BUSINESSES[city], n)

def location(user_id):
    """Finds the longitude and latitude of a user based on the locations of previously-reviewed businesses"""
    
    # init variables
    long = []
    lat = []

    # go over all reviews in all cities
    for city in CITIES:
        for i in REVIEWS[city]:
            # if the review was left by the desired user, save the business_id
            if i["user_id"] == user_id:
                business_id = i["business_id"]
                # go through all businesses and find their exact location
                for j in BUSINESSES[city]:
                    if j["business_id"] == business_id:
                        long.append(j["longitude"])
                        lat.append(j["latitude"])
    
    return {"latitude" : sum(lat)/len(lat), "longitude" : sum(long)/len(long)}

print(location("1-aIDjs2uWYLcoPTDBy5kg"))