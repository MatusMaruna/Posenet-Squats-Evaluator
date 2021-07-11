import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class FrameQualityRecognizerCreator:
    def __init__(self, config):
        self._config = config

    def get_feature_datasets(self):
        train_names, valid_names, test_names = self._get_split_datasets(
            self._config.dataset.path,
            valid_size=self._config.dataset.validation_portion,
            test_size=self._config.dataset.test_portion)

        # Get train, validation and test data
        train_frames, train_numeric, train_labels = self._process_dataset_portion(train_names, is_augmented=True)
        valid_frames, valid_numeric, valid_labels = self._process_dataset_portion(valid_names, is_augmented=False)
        test_frames, test_numeric, test_labels = self._process_dataset_portion(test_names, is_augmented=False)

        train_numeric, valid_numeric, test_numeric = self._process_features(train_numeric, valid_numeric, test_numeric)
        train_labels, valid_labels, test_labels = self._process_labels(train_labels, valid_labels, test_labels)

        return [train_frames, train_numeric], train_labels,\
               [valid_frames, valid_numeric], valid_labels,\
               [test_frames, test_numeric], test_labels

    def _process_dataset_portion(self, names, is_augmented=False):
        frames = []
        numeric = []
        labels = []

        for name in names:
            frm, num, lab = self._get_data_instance(name, self._config.dataset.path,
                                                    is_augmented=is_augmented)
            frames += frm
            numeric += num
            labels += lab

        frames = np.array(frames)
        numeric = np.array(numeric)
        labels = np.array(labels)
        return frames, numeric, labels

    def _process_features(self, train_features, val_features, test_features):
        seq_shape = train_features.shape[1:]
        train_features = train_features.reshape(
            (train_features.shape[0] * train_features.shape[1], train_features.shape[2]))
        scaler = MinMaxScaler()
        train_features = scaler.fit_transform(train_features)
        train_features = train_features.reshape((-1, seq_shape[0], train_features.shape[1]))

        val_features = val_features.reshape(
            (val_features.shape[0] * val_features.shape[1], val_features.shape[2]))
        val_features = scaler.transform(val_features)
        val_features = val_features.reshape((-1, seq_shape[0], val_features.shape[1]))

        test_features = test_features.reshape(
            (test_features.shape[0] * test_features.shape[1], test_features.shape[2]))
        test_features = scaler.transform(test_features)
        test_features = test_features.reshape((-1, seq_shape[0], test_features.shape[1]))

        with open(self._config.model.feature_processor_path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        return train_features, val_features, test_features

    def _process_labels(self, train_labels, val_labels, test_labels):
        scaler = MinMaxScaler()
        train_labels = scaler.fit_transform(train_labels)
        val_labels = scaler.transform(val_labels)
        test_labels = scaler.transform(test_labels)
        with open(self._config.model.label_processor_path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        return train_labels, val_labels, test_labels

    def _get_split_datasets(self, data_dir, test_size=0.1, valid_size=0.2, random_state=11):
        mark_up_csv = f"{data_dir}/1-180.csv"
        mark_up = pd.read_csv(mark_up_csv)
        video_names = list(mark_up["video_name"].values)

        label_dir = f"{data_dir}/labels"
        video_label_names = [n.split(".")[0] for n in video_names]
        video_labels = np.array([np.load(f"{label_dir}/{l}.npy") for l in video_label_names])[:, 0]
        valid_coef = (1.0 / (1.0 - test_size)) * valid_size

        train_names, test_names, y_train, y_test = train_test_split(
            video_label_names, video_labels, test_size=test_size,
            random_state=random_state, stratify=video_labels)

        train_names, valid_names, y_train, y_valid = train_test_split(
            train_names, y_train, test_size=valid_coef,
            random_state=random_state, stratify=y_train)

        return train_names, valid_names, test_names

    def _get_data_by_name(self, video_name, data_dir):
        frames = np.load(f"{data_dir}/frames/{video_name}.npy")
        numeric_data = np.load(f"{data_dir}/csvs/{video_name}.npy")
        labels = np.load(f"{data_dir}/labels/{video_name}.npy")
        return frames, numeric_data, labels

    def _get_data_instance(self, video_name, data_dir, is_augmented=False):
        if is_augmented:
            frames = []
            numeric_data = []
            labels = []
            audmented_names = [p.stem for p in Path(f"{data_dir}/labels/").iterdir() if
                               video_name == p.stem.split("_")[0]]
            for aug_name in audmented_names:
                frm, num_d, lab = self._get_data_by_name(aug_name, data_dir)
                frames.append(frm)
                numeric_data.append(num_d)
                labels.append(lab)
        else:
            frames, numeric_data, labels = self._get_data_by_name(video_name, data_dir)
            frames = [frames]
            numeric_data = [numeric_data]
            labels = [labels]
        return frames, numeric_data, labels
