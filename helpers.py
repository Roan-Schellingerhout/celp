from math import sin, cos, sqrt, atan2, radians 
from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import data
import random
from tqdm import tqdm
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import time
import sklearn.metrics.pairwise as pw
import spacy 


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

def distance_2_businesses(business_id, business_id_2):
    """Calculates the distance between two businesses in km"""
    # radius of earth in km 
    R = 6373.0

    try:
        loc1 = location_business(business_id)
        loc2 = location_business(business_id_2)
    except:
        return "Invalid business_id"

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
    df = df.fillna("(No categories listed)")
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


def create_utility_matrix(df):
    """Creates a utility matrix from a DataFrame (which contains at least the columns stars, user_id, and business_id)"""
    return df.pivot(values='stars', columns='user_id', index='business_id')


def split_data(data, d = 0.75):
    """Split data in a training and test set.
    
    Arguments:
    data -- any dataFrame.
    d    -- the fraction of data in the training set
    """
    np.random.seed(seed=5)
    mask_test = np.random.rand(data.shape[0]) < d
    return data[mask_test], data[~mask_test]


def create_similarity_matrix_categories(matrix):
    """Create a similarity matrix based on categories"""
    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    return pd.DataFrame(m3, index = matrix.index, columns = matrix.index)


def similarity_matrix_cosine(utility_matrix):
    """Creates a cosine similarity matrix from a utility matrix"""
    # mean-center the matrix
    matrix = utility_matrix.sub(utility_matrix.mean())
    # create similarity matrix
    return pd.DataFrame(pw.cosine_similarity(matrix.fillna(0)), index = utility_matrix.index, columns = utility_matrix.index)


def spacy_similarity(df):
    nlp = spacy.load("en_core_web_md")
    sim = pd.DataFrame()

    for i in tqdm(df.index):
        bid = df.loc[i]["business_id"]
        business_1_doc = nlp(df.loc[i]["categories"])
        for j in df.index:
            bid2 = df.loc[j]["business_id"]
            if bid == bid2:
                sim.at[bid, bid2] = 1
            else:
                business_2_doc = nlp(df.loc[j]["categories"])
                sim.at[bid, bid2] = business_1_doc.similarity(business_2_doc)
    return sim


def predict_ratings(similarity, utility, to_predict, n):
    """Predicts the predicted rating for the input test data.
    
    Arguments:
    similarity -- a dataFrame that describes the similarity between items
    utility    -- a dataFrame that contains a rating for each user (columns) and each movie (rows). 
                  If a user did not rate an item the value np.nan is assumed. 
    to_predict -- A dataFrame containing at least the columns movieId and userId for which to do the predictions
    """
    # copy input (don't overwrite)
    ratings_test_c = to_predict.copy()
    # apply prediction to each row
    ratings_test_c['predicted rating'] = to_predict.apply(lambda row: predict_ids(similarity, utility, row['user_id'], row['business_id'], n), axis=1)
    return ratings_test_c

### Helper functions for predict_ratings_item_based ###


def predict_ids(similarity, utility, userId, itemId, n):
    # select right series from matrices and compute
    if userId in utility.columns and itemId in similarity.index:
        return predict_vectors(utility.loc[:,userId], similarity[itemId], n)
    return np.nan


def predict_vectors(user_ratings, similarities, n):
    # select only movies actually rated by user
    relevant_ratings = user_ratings.dropna()
    
    # select corresponding similairties
    similarities_s = similarities[relevant_ratings.index]
    
    # select neighborhood
    similarities_s = similarities_s[similarities_s > 0.0]
    relevant_ratings = relevant_ratings[similarities_s.index]
    
    # if there's nothing left return a prediction of 0
    norm = similarities_s.sum()
    if(norm == 0) or len(similarities_s) < n:
        return np.nan
    
    # compute a weighted average (i.e. neighborhood is all) 
    return np.dot(relevant_ratings, similarities_s)/norm


def mse(predicted_ratings):
    """Computes the mean square error between actual ratings and predicted ratings
    
    Arguments:
    predicted_ratings -- a dataFrame containing the columns rating and predicted rating
    """
    diff = predicted_ratings['stars'] - predicted_ratings['predicted rating']
    return (diff**2).mean()


def test_mse(neighborhood_size = 5, filtertype="collaborative filtering"):
    """Tests the mse of predictions based on a given number of neighborhood sizes

    neighborhood_size -- the sizes of neighborhoods between the number and 1 (so 5 tests for neighborhood of length 1, 2, 3, 4, 5)
    filtertype -- the type of similarity you want to test the mse of   
    """
    # init variables
    all_df = json_to_df()
    df = split_data(all_df)
    ut = create_utility_matrix(df[0])
    print(ut)

    if filtertype == "collaborative filtering":
        print("Creating needed variables...")
        sim = similarity_matrix_cosine(ut)
    elif filtertype == "content based":
        print("Creating needed variables...")
        cats = json_to_df_categories()
        fancy_cats = extract_genres(cats)
        ut_cats = pivot_genres(fancy_cats)
        sim = create_similarity_matrix_categories(ut_cats)
    elif filtertype == "spacy":
        print("Creating needed variables...")
        sim = pd.read_msgpack("spacy_similarity.msgpack")
    else:
        print("Please enter a valid filtertype")
        return

    print("Starting calculations...")
    mses = {}
    # test the mse based on the length of the neighborhood
    for i in tqdm(range(1, neighborhood_size + 1)):     
        predictions = predict_ratings(sim, ut, df[1], i).dropna()
        predictions = predictions[predictions["predicted rating"] > 4.5]
        mses[i] = mse(predictions)
    return mses

##################################################################################
def score(business_id):
    
    # make a dataframe from the BUSINESSES json
    df = pd.DataFrame()
    for city in CITIES:
        business = BUSINESSES[city]
        df = df.append(pd.DataFrame.from_dict(json_normalize(reviews), orient='columns'))
    
    # make the data usable
    df_start = extract_genres(df)
    df_genres = pivot_genres(df_start)
    df_genres_sim = create_similarity_matrix_categories(df_genres)
    df_base = df_genres_sim[business_id]
    
    # for every business_id make 3 columns with values to be used for score
    for i in df_base.index:
        business = df.loc[i]
        df_base.at["review count", i] = business["review_count"]
        df_base.at["stars", i] = business["stars"]
        df_base.at["distance", i] = distance_2_businesses(business_id, i)
        df_base.at["score", i] = business["stars"] * 1 + business["review_count"] * 0.5 + business["distance"] * 0.5 + df_genres_sim[business_id].loc[i] * 1
    
    recommend_id = list(df_base.index)
    recommend_id = recommend_id[0,9]

    recommandations = df.loc[recommend_id]

    # missing feature to confert recommandations from df to list of dicts
    
    return recommandations
##################################################################################


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
