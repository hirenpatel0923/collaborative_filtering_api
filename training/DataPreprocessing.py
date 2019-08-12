import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from DataParams import DataParams


class DataPreprocessing():
    def __init__(self):
        self.dataParams = DataParams()
        self.max_movieId = 0
        self.max_userId = 0
        self.k_factor = 128
        self.seed = 140203

        #self.get_max_movieId()
        #self.get_max_userId()

    def get_max_movieId(self):
        '''This method gives the max_movieid so we can reduce every movieid by 1 to start it from 0 index'''
        movie_csv = pd.read_csv(self.dataParams.MOVIE_CSV)
        self.max_movieId = movie_csv['movieId'].drop_duplicates().max()
        return self.max_movieId

    def get_max_userId(self, rating_csv):
        '''This method gives the max_userid so we can reduce every userid by 1 to start it from 0 index'''
        #rating_csv = pd.read_csv(self.dataParams.RATINGS_CSV)
        self.max_userId = rating_csv['userId'].drop_duplicates().max()
        return self.max_userId

    def get_genres_onehot(self):
        max_length = 5
        movie_csv = pd.read_csv(self.dataParams.MOVIE_CSV)
        movie_csv['movieId'].drop_duplicates()
        movie_csv['genres'] = movie_csv['genres'].str.split('|', expand=False)
        movie_csv['genres_onehot'] = ""
        movie_csv['genres_onehot'] = movie_csv['genres_onehot'].astype(object)
        #get classes        
        onehot = MultiLabelBinarizer()
        onehot.fit_transform(movie_csv['genres'].values)
        classes = list(onehot.classes_)

        #embedding_lst = []
        for index, row in movie_csv.iterrows():
            lst = []
            for genre in row['genres']:
                if genre in classes:
                    i = classes.index(genre)
                    lst.append(i)
            
            length = len(lst)
            if length < max_length:
                for i in range(length, max_length):
                    lst.append(-1)
            if length > max_length:
                #print('grater than ',max_length,' genres... at index : ', index)
                for i in range(max_length, length):
                    lst.pop()

            movie_csv.at[index, 'genres_onehot'] = lst

        return movie_csv



    def load_data(self, shuffle=True):
        rating_csv = pd.read_csv(self.dataParams.RATINGS_CSV)
        
        movie_csv = pd.read_csv(self.dataParams.MOVIE_CSV)
        genres_onehot = self.get_genres_onehot()
        joined_dataset = pd.concat([rating_csv, genres_onehot], axis=1, sort=False)
        #movie_csv_genres = movie_csv['genres'].str.split('|', expand=False)

        shuffled_rating_csv = joined_dataset.copy()
        
        self.get_max_movieId()
        self.get_max_userId(rating_csv)

        # shuffle dataset
        if shuffle:
            shuffled_rating_csv = shuffled_rating_csv.sample(frac=1., random_state=self.seed)
        
        return shuffled_rating_csv['userId'].values, shuffled_rating_csv['movieId'].values, shuffled_rating_csv['rating'].values, shuffled_rating_csv['genres_onehot'].values, joined_dataset, movie_csv, rating_csv


