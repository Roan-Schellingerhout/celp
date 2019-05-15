from math import sin, cos, sqrt, atan2, radians 
from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import data
import random
from tqdm import tqdm
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def distance(user_id, business_id):
    """Calculates the distance between two a user and a business in km"""
    # radius of earth in km 
    R = 6373.0

    try:
        loc1 = location_user(user_id)
        loc2 = location_business(business_id)
    except:
        return "Invalid business_id and/or user_id"

    # convert degrees to radians
    lat1 = radians(loc1[0]) 
    lon1 = radians(loc1[1]) 
    lat2 = radians(loc2[0]) 
    lon2 = radians(loc2[1]) 

    a = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2)**2 
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
    

def location_city(city):
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


def location_user(user_id):
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
                # go through all businesses and, if it's the desired business_id, find the exact location
                for j in BUSINESSES[city]:
                    if j["business_id"] == business_id:
                        long.append(j["longitude"])
                        lat.append(j["latitude"])
    
    if len(long) == 0 or len(lat) == 0:
        return None
    # return the average latitude and longitude of the businesses
    return (sum(lat)/len(lat), sum(long)/len(long))


def location_business(business_id):
    """Finds the location of a business based on its id"""
    # go through all businesses in all cities
    for city in CITIES:
        for business in BUSINESSES[city]:
            # return latitude and longitude once the desired business is found
            if business["business_id"] == business_id:
                return (business["latitude"], business["longitude"])
    # if the business is not present, return None
    return None


def get_businesses(user_id):
    """Finds all the businesses a user has reviewed"""
    business = set()

    # find every business that the user has reviewed.
    for city in CITIES:
        for i in REVIEWS[city]:
            if i["user_id"] == user_id:
                business.add(i["business_id"])
    return business
            

def business_city(business_id):
    """Finds the city a business is located in"""

    # look for businesses until the desired one is found, return the city it is in
    for city in CITIES:
        for i in BUSINESSES[city]:
            if i["business_id"] == business_id:
                return city
    # if no business was found, return None
    return None


def json_to_df():
    """Converts all review.jsons to a single DataFrame containing the columns business_id and user_id"""
    df = pd.DataFrame()

    # add each city's DataFrame to the general DataFrame
    for city in CITIES:
        reviews = REVIEWS[city]
        df = df.append(pd.DataFrame.from_dict(json_normalize(reviews), orient='columns'))
    
    # drop repeated user/business reviews and only save the latest one (since that one is most relevant)
    df = df.drop_duplicates(subset=["business_id", "user_id"], keep="last").reset_index()[["business_id", "stars", "user_id"]]
    return df


def json_to_df_categories():
    """Converts all business.jsons to a single DataFrame containing the columns business_id and categories"""
    df = pd.DataFrame()

    # add each city's DataFrame to the general DataFrame
    for city in CITIES:
        business = BUSINESSES[city]
        df = df.append(pd.DataFrame.from_dict(json_normalize(business), orient='columns'))
    
    # drop repeated user/business reviews and only save the latest one (since that one is most relevant)
    df = df.reset_index()[["business_id", "categories"]]
    return df



def extract_genres(raw_df):
    """Create an unfolded categorie dataframe. Unpacks categories seprated by a ',' into seperate rows.

    Arguments:
    raw_df -- a dataFrame containing at least the columns 'business_id' and 'categories' 
              where categories are seprated by ','
    """
    genres_m = raw_df.apply(lambda row: pd.Series([row['business_id']] + row['categories'].lower().split(",")), axis=1)
    stack_genres = genres_m.set_index(0).stack()
    df_stack_genres = stack_genres.to_frame()
    df_stack_genres['business_id'] = stack_genres.index.droplevel(1)
    df_stack_genres.columns = ['categorie', 'business_id']
    return df_stack_genres.reset_index()[['business_id', 'categorie']]
    

def pivot_genres(df):
    """Create a one-hot encoded matrix for genres.
    
    Arguments:
    df -- a dataFrame containing at least the columns 'business_id' and 'categorie'
    
    Output:
    a matrix containing '0' or '1' in each cell.
    1: the business has the categorie
    0: the business does not have the categorie
    """
    return df.pivot_table(index = 'business_id', columns = 'categorie', aggfunc = 'size', fill_value=0)


def create_similarity_matrix_categories(matrix):
    """Create a similarity matrix based on categories"""
    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    return pd.DataFrame(m3, index = matrix.index, columns = matrix.index)


# jeff = json_to_df_categories()
# jeff_2 = extract_genres(jeff)
# skrrt = pivot_genres(jeff_2)
# print(create_similarity_matrix_categories(skrrt))








"""To be used in jupyter notebook

Data based on following cities: clairton, claremont, clark, clarkson, cleveland, cleveland heigh, cleveland height, clevelant heights, sun city, westlake
"""
# neighborhood_len = [i+1 for i in range(10)]
# mse = [1.7564182528686922, 1.3721995028085512, 1.197445599162912, 1.0903723452322052, 1.0330697173998888, 0.9897513505503943, 0.9468272073142202, 0.9267952685422991, 0.9139840329554876, 0.8908426363903116]
# amount_of_predictions = [16566, 12114, 9735, 8194, 7110, 6222, 5558, 4986, 4512, 4100]

# fig, ax1 = plt.subplots()

# ax1.set_xlabel("Minimum length of neighborhood")
# ax1.set_ylabel("Mean squared error")
# ax1.plot(neighborhood_len, mse, color="red")
# mse = mlines.Line2D([], [], color='red', label='Mean squared error')

# ax2 = plt.twinx()

# ax2.set_ylabel("Amount of predictions")
# ax2.plot(neighborhood_len, amount_of_predictions, color="blue")
# ax2.tick_params(axis="y")
# amount = mlines.Line2D([], [], color='blue', label='Amount of predictions')

# fig.tight_layout()
# plt.legend(handles=[mse, amount], loc="upper right")
# plt.show()