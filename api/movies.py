import pandas as pd
from api.database_connection import Database

db_movie = Database('movie')
db_rating = Database('rating')

class Movie:
    def __init__(self):
        self.convert_to_list()

    def convert_to_list(self):
        db_movie.csv['genres'] = db_movie.csv['genres'].str.split('|')

    def get_all_movies(self):
        return db_movie.csv.to_json(orient='records')

    def get_movie_by_genres(self, genre):
        mask = db_movie.csv.genres.apply(lambda x: genre in x)
        return db_movie.csv[mask].to_json(orient='records')

    def get_movie_by_ratings(self):
        group_by_movie_rating = db_rating.csv.groupby(['movieId']).mean()
        outer_join = pd.concat([db_movie.csv, group_by_movie_rating], axis=1, join_axes=[db_movie.csv.index])
        return outer_join.to_json(orient='records')

    def get_movie_by_genres_ratings(self):
        pass
