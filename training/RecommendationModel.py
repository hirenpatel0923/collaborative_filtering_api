import keras
from keras.models import Model, Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Concatenate, Input
from keras.layers.merge import Dot

class RecommendationModel():
    def __init__(self, num_users, num_movies, k_factor):
        self.num_users = num_users
        self.num_movies = num_movies
        self.k_factor = k_factor
        self.deep_model = None
        self.embeded_model = None

    def userEmbeddingModel(self):
        model = Sequential()
        model.add(Embedding(self.num_users, self.k_factor, input_length=1))
        model.add(Reshape((self.k_factor,)))
        model.compile(optimizer='adam',
                      loss='mse')
        return model

    def movieEmbeddingModel(self):
        model = Sequential()
        model.add(Embedding(self.num_movies, self.k_factor, input_length=1))
        model.add(Reshape((self.k_factor,)))
        model.compile(optimizer='adam',
                      loss='mse')
        return model

    def generate_deepModel(self):
        userModel = self.userEmbeddingModel()
        movieModel = self.movieEmbeddingModel()
        concatenate = Concatenate(axis=-1)([userModel.output, movieModel.output])
        x = Dense(192, activation='relu')(concatenate)
        x = Dropout(0.2)(x)
        x = Dense(self.k_factor, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(1, activation='linear')(x)
        
        model = Model([userModel.input, movieModel.input], x)
        print(model.summary())
        self.deep_model = model
        return model

    def generate_embeddedModel(self):
        userModel = self.userEmbeddingModel()
        movieModel = self.movieEmbeddingModel()
        x = Dot(axes=-1)([userModel.output, movieModel.output])
   
        model = Model([userModel.input, movieModel.input], x)
        print(model.summary())
        self.deep_model = model
        return model
