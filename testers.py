import helpers
import pandas as pd

def test_mse(neighborhood_size = 5, filtertype="collaborative filtering"):
    """Tests the mse of predictions based on a given number of neighborhood sizes

    neighborhood_size -- the sizes of neighborhoods between the number and 1 (so 5 tests for neighborhood of length 1, 2, 3, 4, 5)
    filtertype -- the type of similarity you want to test the mse of   
    """
    # init variables
    all_df = helpers.json_to_df()
    df = helpers.split_data(all_df)
    ut = helpers.create_utility_matrix(df[0])

    if filtertype == "collaborative filtering":
        print("Creating needed variables...")
        sim = helpers.similarity_matrix_cosine(ut)
    elif filtertype == "content based":
        print("Creating needed variables...")
        cats = helpers.json_to_df_categories()
        fancy_cats = helpers.extract_genres(cats)
        ut_cats = helpers.pivot_genres(fancy_cats)
        sim = helpers.create_similarity_matrix_categories(ut_cats)
    elif filtertype == "spacy":
        print("Creating needed variables...")
        sim = pd.read_msgpack("spacy_similarity.msgpack")
    else:
        print("Please enter a valid filtertype")
        return

    print("Starting calculations...")
    mses = {}
    # test the mse based on the length of the neighborhood
    for i in range(1, neighborhood_size + 1):     
        predictions = helpers.predict_ratings(sim, ut, df[1], i).dropna()
        amount = len(predictions)
        mses[i] = helpers.mse(predictions)
    return mses, amount


def predict_all():
    """Fills an entire test set with predictions"""
    
    mses = []
    
    # predict cf based
    all_df = helpers.json_to_df()
    df = helpers.split_data(all_df)
    ut = helpers.create_utility_matrix(df[0])
    sim = helpers.similarity_matrix_cosine(ut)
    predictions = helpers.predict_ratings(sim, ut, df[1], 0)
    mses.append(helpers.mse(predictions))

    # find which values are still np.nan
    to_predict = predictions.loc[~predictions.index.isin(predictions.dropna().index)]
    
    # predict content-based (normal) for those rows
    cats = helpers.json_to_df_categories()
    fancy_cats = helpers.extract_genres(cats)
    ut_cats = helpers.pivot_genres(fancy_cats)
    sim = helpers.create_similarity_matrix_categories(ut_cats)
    
    predictions = predictions.append(helpers.predict_ratings(sim, ut, to_predict, 0))
    mses.append(helpers.mse(predictions))

    # find which values are still np.nan
    to_predict = predictions.loc[~predictions.index.isin(predictions.dropna().index)]

    # predict content-based (spacy) for those rows
    sim = pd.read_msgpack("spacy_similarity.msgpack")
    predictions = predictions.append(helpers.predict_ratings(sim, ut, to_predict, 0))
    to_predict = predictions.loc[~predictions.index.isin(predictions.dropna().index)]
    mses.append(helpers.mse(predictions))

    # for the rows which have no neighborhood in any of the methods, predict the average rating of the test set
    predictions = predictions.fillna(predictions["stars"].mean())
    mses.append(helpers.mse(predictions))

    return mses


"""To be used in jupyter notebook

Data based on following cities: clairton, claremont, clark, clarkson, cleveland, cleveland heigh, cleveland height, clevelant heights, sun city, westlake
"""
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines

# neighborhood_len = [i+1 for i in range(10)]
# mse = [1.7564182528686922, 1.3721995028085512, 1.197445599162912, 1.0903723452322052, 1.0330697173998888, 0.9897513505503943, 0.9468272073142202, 0.9267952685422991, 0.9139840329554876, 0.8908426363903116]
# mse_2 = [1.8679413022514073, 1.4882743924604758, 1.3305999730163216, 1.235784939493885, 1.1872344529077588, 1.1456269013936535, 1.1165820327106752, 1.0889028676424142, 1.0717952556124921, 1.0582349169023897]
# mse_3 = [2.3441104860640687, 1.79612055000979, 1.5573728644495473, 1.4420303560817995, 1.2950954895714613, 1.2682969141074594, 1.2662363564547128, 1.2908763865812998, 1.273829947371037, 1.3300483826754819]

# plt.xlabel("Minimum length of neighborhood")
# plt.ylabel("Mean squared error")
# plt.plot(neighborhood_len, mse, color="red")
# mse_cf = mlines.Line2D([], [], color='red', label='mse (item-based collaborative filtering)')

# plt.plot(neighborhood_len, mse_2, color="blue")
# mse_cb = mlines.Line2D([], [], color='blue', label='mse (content-based filtering)')

# plt.plot(neighborhood_len, mse_3, color="green")
# mse_cb_spacy = mlines.Line2D([], [], color='green', label='mse (content-based filtering with spacy)')


# fig.tight_layout()
# plt.legend(handles=[mse, amount], loc="upper right")
# plt.show()