import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ExerciseQualityDataCreator:
    def __init__(self, config):
        self._config = config

    def get_feature_datasets(self):
        train_names, valid_names, test_names = self._get_split_datasets(
            self._config.dataset.path,
            valid_size=self._config.dataset.validation_portion,
            test_size=self._config.dataset.test_portion
        )

        x_train, y_train = self._process_dataset_portion(train_names, is_augmented=True)
        x_val, y_val = self._process_dataset_portion(valid_names, is_augmented=False)
        x_test, y_test = self._process_dataset_portion(test_names, is_augmented=False)

        # Scale the data
        x_train, x_val, x_test = self._process_features(x_train, x_val, x_test)
        return x_train, x_test, y_train, y_test, x_val, y_val

    def _process_features(self, train_features, val_features, test_features):
        seq_shape = train_features.shape[1:]
        train_features = train_features.reshape(
            (train_features.shape[0] * train_features.shape[1], train_features.shape[2]))
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        train_features = train_features.reshape((-1, seq_shape[0], train_features.shape[1], 1))

        val_features = val_features.reshape(
            (val_features.shape[0] * val_features.shape[1], val_features.shape[2]))
        val_features = scaler.transform(val_features)
        val_features = val_features.reshape((-1, seq_shape[0], val_features.shape[1], 1))

        test_features = test_features.reshape(
            (test_features.shape[0] * test_features.shape[1], test_features.shape[2]))
        test_features = scaler.transform(test_features)
        test_features = test_features.reshape((-1, seq_shape[0], test_features.shape[1], 1))

        with open(self._config.model.feature_processor_path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        return train_features, val_features, test_features

    def _process_dataset_portion(self, names, is_augmented=False):
        features = []
        labels = []
        for name in names:
            data, lab = self._get_data_instance(name, self._config.dataset.path, is_augmented=is_augmented)
            features += data
            labels += lab
        features = np.array(features)
        labels = np.array(labels)
        return features, labels

    def _get_split_datasets(self, data_dir, valid_size=0.2, test_size=0.1, random_state=11):
        all_names = [f.stem.split("_")[0] for f in Path(f"{data_dir}").iterdir()]
        files = set(all_names)
        names = list(files)
        labels = [1 if n[0] == "G" else 0 for n in names]
        valid_coef = (1.0 / (1.0 - test_size)) * valid_size

        train_names, test_names, y_train, y_test = train_test_split(
            names, labels, test_size=test_size,
            random_state=random_state, stratify=labels)

        train_names, valid_names, y_train, y_valid = train_test_split(
            train_names, y_train, test_size=valid_coef,
            random_state=random_state, stratify=y_train)
        return train_names, valid_names, test_names
    
    def _get_data_by_name(self, record_name, data_dir):
        features = pd.read_csv(f"{data_dir}/{record_name}.csv")
        label = 1 if record_name[0] == "G" else 0
        return features, label

    def _get_data_instance(self, record_name, data_dir, is_augmented=True):
        data = []
        labels = []
        if is_augmented:
            audmented_names = [p.stem for p in Path(f"{data_dir}").iterdir() if record_name == p.stem.split("_")[0]]
            for aug_n in audmented_names:
                d, lab = self._get_data_by_name(aug_n, data_dir)
                data.append(d.values)
                labels.append(lab)
        else:
            d, lab = self._get_data_by_name(record_name, data_dir)
            data.append(d.values)
            labels.append(lab)
        return data, labels
