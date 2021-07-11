import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class KeyFramesDetectorDataCreator:
    def __init__(self, config):
        self._config = config
        self._label_padding = 4

    def get_feature_datasets(self):
        data = pd.DataFrame()
        for csv_posix in Path(self._config.dataset.path).iterdir():
            if csv_posix.suffix != ".csv":
                continue
            d = pd.read_csv(str(csv_posix))

            d.loc[:self._label_padding, "Timeline_OneHot"] = [
                d.loc[0, "Timeline_OneHot"] for _ in range(self._label_padding + 1)]
            d.loc[len(d) - (self._label_padding + 1):len(d), "Timeline_OneHot"] = [
                d.loc[len(d) - 1, "Timeline_OneHot"] for _ in range(self._label_padding + 1)]

            data = pd.concat([data, d], axis=0)

        data = data.drop('Timeline', axis=1)

        raw_labels = np.array(data.pop('Timeline_OneHot'))
        labels = np.array([np.fromstring(v[1:-1], dtype=int, sep=' ') for v in raw_labels])
        raw_features = np.array(data)

        class_weights = self._get_class_weight(labels)
        features = self._process_features(raw_features)

        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=self._config.dataset.test_portion,
            random_state=self._config.dataset.random_state, stratify=labels)
        return x_train, x_test, y_train, y_test, class_weights

    def _process_features(self, raw_features):
        scaler = StandardScaler()
        features = scaler.fit_transform(raw_features)
        features = np.clip(features, -5, 5)
        with open(self._config.model.feature_processor_path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        return features

    def _get_class_weight(self, labels):
        total = len(labels)
        class_1_num = sum(labels[:, 0])
        class_2_num = sum(labels[:, 1])
        class_3_num = sum(labels[:, 2])

        class_weights = {0: (total / class_1_num) / 2.0,
                         1: (total / class_2_num) * 2.0,
                         2: (total / class_3_num) * 2.0,
                         }
        return class_weights
