from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import data
import random
from tqdm import tqdm
import helpers
import pandas as pd
import numpy as np
import time
import sklearn.metrics.pairwise as pw
from pandas.io.json import json_normalize
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


def filter_recommendations(user_id, recommendations, n):
    """Filters recommendations down to nearby, open businesses"""
    # init variables
    out_of_businesses = set()

    # find all businesses that are out of business
    for city in CITIES:
        for business in BUSINESSES[city]:
            if not business["is_open"]:
                out_of_businesses.add(business["business_id"])

    # drop all recommendations of business that are out of business
    for business in recommendations:
        if business["business_id"] in out_of_businesses:
            recommendations.remove(business)

    # remove businesses that are too far away, as long as enough recommendations remain
    for business in recommendations:
        if helpers.distance(user_id, business["business_id"]) > 25 and len(recommendations) > n:
            recommendations.remove(business)
    return recommendations


def create_utility_matrix(df):
    """Creates a utility matrix from a DataFrame (which contains at least the columns stars, user_id, and business_id)"""
    return df.pivot(values='stars', columns='user_id', index='business_id')


def similarity_matrix_cosine(utility_matrix):
    """Creates a cosine similarity matrix from a utility matrix"""
    # mean-center the matrix
    matrix = utility_matrix.sub(utility_matrix.mean()).fillna(0)
    # create similarity matrix
    return pd.DataFrame(pw.cosine_similarity(matrix), index = utility_matrix.index, columns = utility_matrix.index)





"""probably throwaway code"""
# def create_utility_matrix():
#     """Create a utility matrix of businesses and user reviews"""
#     # create an empty DataFrame
#     utility_matrix = pd.DataFrame()

#     # go over all cities
#     for city in CITIES:
#         # create a city-specific DataFrame
#         cityFrame = pd.DataFrame()
#         # add all reviews from that city to the DataFrame
#         for review in tqdm(REVIEWS[city]):
#             cityFrame.at[review["business_id"], review["user_id"]] = review["stars"]
#         # append the city DataFrame to the utility matrix
#         utility_matrix = utility_matrix.append(cityFrame, sort=False)
    
#     return utility_matrix