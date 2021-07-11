from keras import optimizers
from keras import Sequential
from keras.layers import Dense, InputLayer, BatchNormalization, Activation
from keras.losses import mean_absolute_error


class Pose2KinectModel:
    def __init__(self, config):
        self._config = config
        self._optimizer = optimizers.Adam(lr=self._config.training.learning_rate)
        self.model = self.build_model()

    def compile(self):
        self.model.compile(optimizer=self._optimizer, loss=mean_absolute_error)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        return self.model.load_weights(file_path)

    def predict(self, features):
        return self.model.predict(features)

    def build_model(self):
        model = Sequential([
            InputLayer(input_shape=(26,)),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(26)
        ])
        return model
