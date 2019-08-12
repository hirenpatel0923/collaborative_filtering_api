import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

#from keras.callbacks import ModelCheckpoint

from RecommendationModel import RecommendationModel
from DataPreprocessing import DataPreprocessing

print('script running.............')


dataPreprocessing = DataPreprocessing()

userIds, movieIds, ratings, genres, joined_dataset, movie_csv, rating_csv = dataPreprocessing.load_data()

print(len(genres))
rModel = RecommendationModel(dataPreprocessing.max_userId, dataPreprocessing.max_movieId, dataPreprocessing.k_factor)
model = rModel.generate_embeddedModel()

model.load_weights('weights_best_embedded.hdf5')


def predict_rating(userid, movieid):
    return model.predict([np.array([userId]), np.array([movieid])])[0][0]


userId = 50000
user_movies = [
    [userId,1234.0,4.0],
    [userId,3421.0,3.0],
    [userId,2341.0,2.0],
    [userId,142.0,3.2],
    [userId,4214.0,5]
]
columns = ['userId','movieId','rating']

#ratings = pd.read_csv('../data/rating.csv')

user_ratings = pd.DataFrame(data=user_movies, columns=columns)

user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(userId, x['movieId']), axis=1)

print(user_ratings.sort_values(by='rating', 
                         ascending=False).merge(movie_csv, 
                                                on='movieId', 
                                                how='inner', 
                                                suffixes=['_u', '_m']).head(10))

print('prediction..............')

recommendations = rating_csv[rating_csv['movieId'].isin(user_ratings['movieId']) == False][['movieId']].drop_duplicates()

recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(userId, x['movieId']), axis=1)

print(recommendations.sort_values(by='prediction',
                          ascending=False).merge(movie_csv,
                                                 on='movieId',
                                                 how='inner',
                                                 suffixes=['_u', '_m']).head(10))

