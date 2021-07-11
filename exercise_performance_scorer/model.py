from keras import Sequential, optimizers
from keras.layers import Dense, Flatten, GRU, InputLayer
from keras.losses import mean_squared_error


class ExercisePerformanceModel:
    def __init__(self, config):
        self._config = config
        self._optimizer = optimizers.Adam(lr=self._config.training.learning_rate)
        self.model = self.build_model()


    def compile(self):
        self.model.compile(optimizer=self._optimizer, loss=mean_squared_error)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        return self.model.load_weights(file_path)

    def predict(self, features):
        return self.model.predict(features)

    def build_model(self):
        model = Sequential()
        model.add(InputLayer((20, 39)))
        model.add(GRU(64, return_sequences=True))
        model.add(GRU(32, return_sequences=True))
        model.add(GRU(8, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(1))
        return model
