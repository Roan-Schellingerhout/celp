import helpers
from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import data
import random
import pandas as pd


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
    if not user_id and not business_id:
        city = random.choice(CITIES)
        return random.sample(BUSINESSES[city], n)


    elif user_id and not business_id:
        prediction = predict_for_user(user_id=user_id, n=n).nlargest(n,'predicted rating')
        prediction = filter_recommendations(user_id, prediction, n)
        prediction = prediction.drop(['user_id', "predicted rating"], axis=1)
        
        # insert cities of the businesses in the dataframe
        prediction['stars']=None
        prediction['name']=None
        prediction['city']=None
        prediction['adress']=None
        for x in prediction.index:
            prediction.at[x,'city']=helpers.business_city(prediction.loc[x]["business_id"])
        
        # insert stars, names, and address of the businesses in the dataframe
        for x in prediction.index:
            business_data=helpers.get_business(prediction.loc[x]["business_id"])
            prediction.at[x,'stars']=business_data[0]
            prediction.at[x,'name']=business_data[1]
            prediction.at[x,'adress']=business_data[2]
        
        # clean the dataframe
        prediction = prediction.applymap(str)
        prediction = prediction.drop("index", axis=1)
        
        return prediction.to_dict(orient='records')

    elif business_id:
        # init variables
        prediction = pd.DataFrame()
        similar_businesses = helpers.score(business_id, n=10)
        
        # find the city of each recommendation
        for x in similar_businesses:
            prediction.at[x,'city']=helpers.business_city(x)
        
        # insert stars, names, and address of the businesses in the dataframe
        for x in prediction.index:
            business_data=helpers.get_business(x)
            prediction.at[x,'stars']=business_data[0]
            prediction.at[x,'name']=business_data[1]
            prediction.at[x,'adress']=business_data[2]
        
        # clean the dataframe
        prediction = prediction.applymap(str)

        return prediction.to_dict(orient='records')


def predict_for_user(user_id=None, n=10):
    """predict as many ratings as possible for a given user"""

    # create a DataFrame containing the necessary information
    all_df = helpers.json_to_df()
    
    # find all businesses reviewed by the user
    reviewed = all_df.loc[all_df["user_id"] == user_id]["business_id"].values
    # find all unique businesses
    all_businesses = all_df["business_id"].unique()
    
    # create a DataFrame on which to do the predictions
    to_predict = pd.DataFrame(columns=["user_id", "business_id"])
    to_predict["business_id"] = all_businesses
    to_predict["user_id"] = user_id
    to_predict = to_predict[~to_predict["business_id"].isin(reviewed)].reset_index()   

    # predict using item-based collaborative filtering
    ut = helpers.create_utility_matrix(all_df)
    sim = helpers.similarity_matrix_cosine(ut)
    predictions = helpers.predict_ratings(sim, ut, to_predict, 0)[["user_id", "business_id", "predicted rating"]]
    
    # predict using content-based filtering
    cats = helpers.json_to_df_categories()
    fancy_cats = helpers.extract_genres(cats)
    ut_cats = helpers.pivot_genres(fancy_cats)
    sim = helpers.create_similarity_matrix_categories(ut_cats)
    predictions_2 = helpers.predict_ratings(sim, ut, to_predict, 0)[["user_id", "business_id", "predicted rating"]]
    
    # find the new predictions made and add them to the df of predictions
    new_predictions = predictions_2[~predictions_2["business_id"].isin(predictions.dropna()["business_id"].values)]
    predictions = predictions.append(new_predictions)

    # predict using spaCy-based content-based filtering
    sim = pd.read_msgpack("spacy_similarity.msgpack")
    predictions_3 = helpers.predict_ratings(sim, ut, to_predict, 0, cutoff=0.8)[["user_id", "business_id", "predicted rating"]]
    
    # find the new predictions made and add them to the df of predictions
    new_predictions = predictions_3[~predictions_3["business_id"].isin(predictions.dropna()["business_id"].values)]      
    predictions = predictions.append(new_predictions)

    predictions = predictions.dropna().reset_index()[["user_id", "business_id", "predicted rating"]]

    # if not enough predictions could be made, add random highly rated businesses to the predictions
    while len(predictions) < n:
        insert_random = random.choice(all_df[all_df["stars"] > 4]["business_id"].values)
        if insert_random not in predictions["business_id"].values:
            predictions.loc[predictions.index.max() + 1] = [user_id, insert_random, predictions["predicted rating"].mean()]

    return predictions


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
        if recommendations.loc[i]["business_id"] in out_of_businesses and len(recommendations) > n:
            recommendations = recommendations.drop(index=i)

    # remove businesses that are too far away, as long as enough recommendations remain
    for i in recommendations.index:
        if helpers.distance(user_id, recommendations.loc[i]["business_id"]) > 25 and len(recommendations) > n:
            recommendations = recommendations.drop(index=i)
    return recommendations.reset_index()
