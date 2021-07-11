import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class KinectDataCreator:
    def __init__(self, config):
        self._config = config

    def get_feature_datasets(self):
        data = pd.DataFrame()
        for csv_posix in Path(self._config.dataset.path).iterdir():
            if csv_posix.suffix != ".csv":
                continue
            d = pd.read_csv(str(csv_posix))
            data = pd.concat([data, d], axis=0)
        data = data.sample(frac=1).reset_index(drop=True)
        shuffled_data = data.sample(n=(data.shape[0]), random_state=self._config.dataset.random_state)
        shuffled_data.drop('FrameNo', axis=1, inplace=True)

        inputs = shuffled_data.filter(regex='_x|_y')
        labels = shuffled_data.filter(regex='_z')
        inputs = self._process_features(inputs)
        labels = self._process_labels(labels)

        x_train, x_test, y_train, y_test = train_test_split(
            inputs, labels, test_size=self._config.dataset.test_portion,
            random_state=self._config.dataset.random_state)
        return x_train, x_test, y_train, y_test

    def _process_features(self, raw_features):
        scaler = MinMaxScaler()
        features = scaler.fit_transform(raw_features)
        with open(self._config.model.feature_processor_path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        return features

    def _process_labels(self, raw_labels):
        scaler = MinMaxScaler()
        labels = scaler.fit_transform(raw_labels)
        with open(self._config.model.label_processor_path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        return labels
