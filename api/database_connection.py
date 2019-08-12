import pandas as pd

class Database:
    def __init__(self):
        #if table == 'movie':
        self.MovieCsv = pd.read_csv('./data/movie.csv')
        #if table == 'link':
        self.LinkCsv = pd.read_csv('./data/link.csv')
        #if table == 'rating':
        #self.csv = pd.read_csv('./data/rating.csv')
    