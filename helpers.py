def woonplaatsen():
    # init variables
    woonplaatsen = {}

    # find the city of each user
    for city in CITIES:
        for i in USERS[city]:
            woonplaatsen[i["user_id"]] = city
    return woonplaatsen
    

def locatie():
    # init variables
    long = []
    lat = []
    locaties = {}

    # for each city, calculate the average longitude and latitude
    for city in CITIES:
        for i in BUSINESSES[city]:
            long.append(i["longitude"])
            lat.append(i["latitude"])
        if long and lat:
            locaties[city] = (sum(lat)/len(lat), sum(long)/len(long))
        else:
            locaties[city] = None
        long = []
        lat = []
    return locaties



# def location(user_id):
#     """Finds the longitude and latitude of a user based on the locations of previously-reviewed businesses"""
    
#     # init variables
#     long = []
#     lat = []

#     # go over all reviews in all cities
#     for city in tqdm(CITIES):
#         print("a")
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