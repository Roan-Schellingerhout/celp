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
    for i in recommendations.index:
        if recommendations.loc[i]["business_id"] in out_of_businesses:
            recommendations = recommendations.drop(index=i)

    # remove businesses that are too far away, as long as enough recommendations remain
    for i in recommendations:
        if helpers.distance(user_id, recommendations.loc[i]["business_id"]) > 25 and len(recommendations) > n:
            recommendations = recommendations.drop(index=i)
    return recommendations.reset_index()



def create_utility_matrix(df):
    """Creates a utility matrix from a DataFrame (which contains at least the columns stars, user_id, and business_id)"""
    return df.pivot(values='stars', columns='user_id', index='business_id')



def similarity_matrix_cosine(utility_matrix):
    """Creates a cosine similarity matrix from a utility matrix"""
    # mean-center the matrix
    matrix = utility_matrix.sub(utility_matrix.mean())
    # create similarity matrix
    return pd.DataFrame(pw.cosine_similarity(matrix.fillna(0)), index = utility_matrix.index, columns = utility_matrix.index)


def mse(predicted_ratings):
    """Computes the mean square error between actual ratings and predicted ratings
    
    Arguments:
    predicted_ratings -- a dataFrame containing the columns rating and predicted rating
    """
    diff = predicted_ratings['stars'] - predicted_ratings['predicted rating']
    return (diff**2).mean()


def split_data(data, d = 0.75):
    """Split data in a training and test set.
    
    Arguments:
    data -- any dataFrame.
    d    -- the fraction of data in the training set
    """
    np.random.seed(seed=5)
    mask_test = np.random.rand(data.shape[0]) < d
    return data[mask_test], data[~mask_test]



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


def test_mse(neighborhood_size = 5, sample = 25, repetitions = 0):
    """Tests the mse of predictions based on a given number of neighborhood sizes

    -neighborhood_size: the sizes of neighborhoods between the number and 1 (so 5 tests for neighborhood of length 1, 2, 3, 4, 5)
    -sample: the amount of indices of the predictions to be taken into consideration (useful for making sure the amount of predictions don't matter)
    -repetitions: the amount of times the experiment should be done, more repetitions give more accurate results
    
    """
    # init variables
    all_df = json_to_df()
    df = split_data(all_df)
    ut = create_utility_matrix(df[0])
    sim = similarity_matrix_cosine(ut)

    # test the mse based on the length of the neighborhood
    for i in range(1, neighborhood_size+1):
        # if more than 0 repetitions are required, repeat the process
        if repetitions:
            rep_loop = []
            for j in range(repetitions):
                predictions = predict_ratings(sim, ut, df[1], i).dropna()
                indices = sorted(random.choices(predictions.index, k=sample))
                rep_loop.append(mse(predictions.loc[indices]))
            print("The mse for a neighborhood of length", i, "with a sample size of", sample, "over", repetitions, "repetitions is", sum(rep_loop)/len(rep_loop))
            rep_loop = []
        # if no repetitions are required, calculate the mse's once
        else:
            predictions = predict_ratings(sim, ut, df[1], i).dropna()
            indices = sorted(random.choices(predictions.index, k=sample))
            print("The mse for a neighborhood of length", i, "with a sample size of", sample, "is", mse(predictions.loc[indices]))

test_mse(repetitions = 5)