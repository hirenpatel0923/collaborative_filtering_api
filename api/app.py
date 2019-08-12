from flask import json, request, Blueprint, Flask
import os
import ast
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from RecommendationModel import RecommendationModel
from DataPreprocessing import DataPreprocessing

app = Flask(__name__)

#mod = Blueprint('API', __name__)

dataPreprocessing = DataPreprocessing()
movie_csv, link_csv, userIds_csv = dataPreprocessing.load_data()

rModel = RecommendationModel(dataPreprocessing.max_userId, dataPreprocessing.max_movieId, dataPreprocessing.k_factor)
model = rModel.generate_embeddedModel()

model.load_weights('weights_best_embedded.hdf5')

global graph
graph = tf.get_default_graph()


def predict_rating(model, userid, movieid):
    return model.predict([np.array([userid]), np.array([movieid])])[0][0]

@app.route('/')
def hello_world():
    return 'Hello World...'

@app.route('/test', methods=['post'])
def testAPI():
    if request.method == 'POST':
        with graph.as_default():
            json_dict = request.get_json()
            lists = ast.literal_eval(json_dict['list'])
            row = userIds_csv.sample(n=1)
            userid = row['1'].values


            columns = ['userId','movieId','rating']

            ratings = np.array(lists)

            id = np.full((ratings.shape[0], 1), userid[0], dtype=int)
            
            user_ratings = np.hstack((id, ratings))
            
            user_df = pd.DataFrame(data=user_ratings, columns=columns)
            
            user_df['userId'] = user_df['userId'].astype(np.int64)
            user_df['movieId'] = user_df['movieId'].astype(np.int64)
            #user_df = user_df.astype({"userId":int, "movieId":int})
            
            user_df['prediction'] = user_df.apply(lambda x: predict_rating(model, x['userId'], x['movieId']), axis=1)

            recommendations = movie_csv[movie_csv['movieId'].isin(user_df['movieId']) == False][['movieId']].drop_duplicates()
            recommendations['movieId'] = recommendations['movieId'].apply(lambda x: x - 1)
            recommendations['userId'] = userid[0]
            recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(model, x['userId'], x['movieId']), axis=1)
            
            recommendations = recommendations.sort_values(by='prediction',
                            ascending=False).merge(movie_csv,
                                                    on='movieId',
                                                    how='inner',
                                                    suffixes=['_u', '_m'])
            
            recommendations = recommendations.sort_values(by='prediction',
                            ascending=False).merge(link_csv,
                                                    on='movieId',
                                                    how='inner',
                                                    suffixes=['_u', '_m'])
            temp_df = recommendations.head(10)
            temp_df = temp_df['tmdbId']
            return temp_df.to_json()
        #user_ratings = pd.DataFrame(data=json_dict['list'])


if __name__ == "__main__":
    #app.run()
    app.run(host="0.0.0.0", port=8888)

#movie object
# movie = Movie()

# @mod.route('/all_movies')
# def getAllMovies():
#     return movie.get_all_movies()

# @mod.route('/movie_by_ratings')
# def getByRatings():
#     return movie.get_movie_by_ratings()

# @mod.route('/movie_by_genre', methods=['get'])
# def getByGeneres():
#     genre = request.args.get('genre')
#     return movie.get_movie_by_genres(genre)

