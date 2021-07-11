from keras import optimizers
from keras import Sequential
from keras.metrics import TruePositives, FalsePositives, TrueNegatives, \
    FalseNegatives, BinaryAccuracy, Precision, Recall, AUC
from keras.layers import Dense, Dropout, BatchNormalization


class KeyFramesDetectorModel:
    def __init__(self, config):
        self._config = config
        self._optimizer = optimizers.Adam(lr=self._config.training.learning_rate)
        self.model = self.build_model()

    def compile(self):
        model_metrics = [
            TruePositives(name='tp'),
            FalsePositives(name='fp'),
            TrueNegatives(name='tn'),
            FalseNegatives(name='fn'),
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
        ]
        self.model.compile(optimizer=self._optimizer, loss='categorical_crossentropy', metrics=model_metrics)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        return self.model.load_weights(file_path)

    def predict(self, features):
        return self.model.predict(features)

    def build_model(self):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(39,)),
            BatchNormalization(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(1024),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        return model
