from keras import optimizers
from keras import Sequential
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
import tensorflow as tf


class ExerciseQualityModel:
    def __init__(self, config):
        self._config = config
        self._optimizer = optimizers.Adam(lr=self._config.training.learning_rate)
        self.model = self.build_model()
        self._METRICS = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]


    def compile(self):
        self.model.compile(optimizer=self._optimizer, loss=binary_crossentropy, metrics = self._METRICS)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        return self.model.load_weights(file_path)

    def predict(self, features):
        return self.model.predict(features)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(8, (1, 1), activation='tanh', input_shape=(10, 39, 1))) # Could be defined better ?
        model.add(MaxPool2D((1, 1), padding='same'))
        model.add(Conv2D(8, (2, 2), activation='tanh'))
        model.add(MaxPool2D((2, 2), padding='same'))
        model.add(Conv2D(16, (3, 3), activation='tanh'))
        model.add(MaxPool2D((3, 3), padding='same'))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model
