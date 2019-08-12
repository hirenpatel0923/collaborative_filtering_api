import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

#from keras.callbacks import ModelCheckpoint

from RecommendationModel import RecommendationModel
from dp import DataPreprocessing

print('script running.............')


dataPreprocessing = DataPreprocessing()

userIds, movieIds, ratings, genres, joined_dataset, movie_csv, rating_csv = dataPreprocessing.load_data()
