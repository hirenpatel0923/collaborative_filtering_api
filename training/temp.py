import pandas as pd
import numpy as np

userId = 50000
movies = [
    [userId,1234,4.0],
    [userId,3421,3.0],
    [userId,2341,2.0],
    [userId,142,3.2],
    [userId,4214,5]
]
columns = ['userid','movieid','rating']

#ratings = pd.read_csv('../data/rating.csv')

user_ratings = pd.DataFrame(data=movies, columns=columns)

print(user_ratings)