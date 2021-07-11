from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, \
    Flatten, Input, GRU, Concatenate, TimeDistributed
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import MAE


class FrameQualityRecognizerModel:
    def __init__(self, config):
        self._config = config
        self._optimizer = optimizers.Adam(lr=self._config.training.learning_rate)
        self.model = self.build_model()

    def compile(self):
        self.model.compile(optimizer=self._optimizer, loss=MAE)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        return self.model.load_weights(file_path)

    def predict(self, features):
        return self.model.predict_classes(features)

    def build_model(self):
        frame_data_model = self.build_frame_data_model()
        numeric_data_model = self.build_numeric_data_model()
        concat = Concatenate(axis=1)
        combined = concat([frame_data_model.output, numeric_data_model.output])
        dense = Dense(1024, activation="relu")(combined)
        dropout = Dropout(0.25)(dense)
        output = Dense(2)(dropout)
        model = Model([frame_data_model.input, numeric_data_model.input], output)
        return model

    def build_frame_data_model(self):
        model = Sequential()
        model.add(Input(shape=(10, 224, 224, 3)))
        model.add(TimeDistributed(Conv2D(128, (2, 2), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(2, 2)))
        model.add(TimeDistributed(Conv2D(64, (2, 2), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(2, 2)))
        model.add(TimeDistributed(Conv2D(32, (2, 2), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(2, 2)))
        model.add(TimeDistributed(Flatten()))
        model.add(GRU(256, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        return model

    def build_numeric_data_model(self):
        model = Sequential()
        model.add(Input(shape=(10, 40)))
        model.add(GRU(128))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        return model
