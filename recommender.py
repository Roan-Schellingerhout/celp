from helpers import *
from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import data
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
    print(predict(user_id=user_id, business_id=business_id).nlargest(10,'predicted rating'))
    return predict(user_id=user_id, business_id=business_id).nlargest(10,'predicted rating')





def predict(user_id=None, business_id=None):
    user_predictions = pd.DataFrame()
    
    all_df = json_to_df()
    ut = create_utility_matrix(all_df)
    asd = ut[user_id].dropna().index
    print(ut.shape)
    ut = ut.drop(index=asd)
    print(ut.shape)
    sim = similarity_matrix_cosine(ut)
    predictions = predict_ratings(sim, ut, all_df, 0)
    return predictions[predictions["user_id"] == user_id]





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
        if distance(user_id, recommendations.loc[i]["business_id"]) > 25 and len(recommendations) > n:
            recommendations = recommendations.drop(index=i)
    return recommendations.reset_index()


recommend(user_id="--kdggJkUFNxLmQvzvYkUg", business_id="-2XMn8phKIqizvss9PBLCw")