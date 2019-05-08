from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import data
import random
from tqdm import tqdm
import helpers
import pandas as pd
import numpy as np
import time
# from pandas_msgpack import to_msgpack, read_msgpack


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


def create_utility_matrix():
    """Create a utility matrix of businesses and user reviews"""
    # init variables
    business_ids = set()
    user_ids = set()

    # find all business_ids
    for city in CITIES:
        for i in BUSINESSES[city]:
            business_ids.add(i["business_id"])

    # find all user_ids
    for city in CITIES:
        for i in USERS[city]:
            user_ids.add(i["user_id"])

    # create a DataFrame with the businesses and users
    utility = pd.DataFrame(np.nan, index = business_ids, columns = user_ids)
    
    # for each user/business combo, if the user has reviewed the business, find the last given rating
    for business in utility.index:
        for user in utility:
            if business in helpers.get_businesses(user):
                utility.at[business, user] = data.get_reviews(helpers.business_city(business), business, user)[-1]["stars"]
    return utility