from keras.callbacks import ModelCheckpoint

from RecommendationModel import RecommendationModel
from DataPreprocessing import DataPreprocessing

dataPreprocessing = DataPreprocessing()
userIds, movieIds, ratings, rating_csv, movie_csv = dataPreprocessing.load_data()


rModel = RecommendationModel(dataPreprocessing.max_userId, dataPreprocessing.max_movieId, dataPreprocessing.k_factor)
model = rModel.generate_embeddedModel()
model.compile(loss='mse',
              optimizer='adamax')
print(model.summary())

############## Training part ################
print('userIds : ',userIds, ' shape : ',userIds.shape)
print('movieIds : ',movieIds, ' shape : ',movieIds.shape)
print('ratings : ',ratings, ' shape : ',ratings.shape)
print('start training...........')
filepath = 'weights_best.hdf5'
checkPoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_lst = [checkPoint]
model.fit([userIds, movieIds], ratings, validation_split=.1, epochs=10, batch_size=512, callbacks=callbacks_lst)



