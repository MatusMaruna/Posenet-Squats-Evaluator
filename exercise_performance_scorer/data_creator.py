import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ExercisePerformanceDataCreator:
    def __init__(self, config):
        self._config = config

    def get_feature_datasets(self):
        data_dir = self._config.dataset.path
        is_kinect = True
        labels = self._config.dataset.scores_path
        train_names, test_names = self._get_split_datasets(data_dir)

        # Get validation set names
        validation_split = 0.1
        slice_indx = int(len(train_names) - len(train_names) * validation_split)
        validation_names = train_names[slice_indx:]
        train_names = train_names[:slice_indx]

        # Get data
        train_features = []
        train_labels = []

        test_features = []
        test_labels = []

        val_features = []
        val_labels = []

        for name in train_names:
            data, labels = self._get_data_instance(name, labels, data_dir, True, is_kinect)
            for i in range(data.shape[0]):
                train_features.append(data[i])
            for i in range(labels.shape[0]):
                train_labels.append(labels[i])

        train_features = np.array(train_features)
        train_labels = np.array(train_labels)

        for name in test_names:
            data, labels = self._get_data_instance(name, labels, data_dir, False, is_kinect)
            for i in range(data.shape[0]):
                test_features.append(data[i])
            test_labels.append(labels)

        test_features = np.array(test_features)
        test_labels = np.array(test_labels)

        for name in validation_names:
            data, labels = self._get_data_instance(name, labels, data_dir, True, is_kinect)
            for i in range(data.shape[0]):
                val_features.append(data[i])
            for i in range(labels.shape[0]):
                val_labels.append(labels[i])

        val_features = np.array(val_features)
        val_labels = np.array(val_labels)

        train_features, val_features, test_features = self._process_features(train_features, val_features, test_features)
        train_labels, val_labels, test_labels = self._process_labels(train_labels, test_labels,val_labels)

        return train_features, train_labels, test_features, test_labels, val_features, val_labels

    def _process_features(self, train_features, val_features, test_features):
        seq_shape = train_features.shape[1:]
        train_features = train_features.reshape(
            (train_features.shape[0] * train_features.shape[1], train_features.shape[2]))
        scaler = StandardScaler()
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

    def _process_labels(self, train_labels, test_labels, val_labels):
        scaler = MinMaxScaler()
        train_labels = scaler.fit_transform(train_labels)
        val_labels = scaler.transform(val_labels)
        test_labels = scaler.transform(test_labels)
        with open(self._config.model.label_processor_path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        return train_labels, val_labels, test_labels

    def _get_split_datasets(self, data_dir, test_size=0.15, random_state=11):
        all_names = [f.stem.split("_")[0] for f in Path(f"{data_dir}").iterdir()]
        files = set(all_names)
        names = list(files)
        labels = np.zeros(len(names))

        train_names, test_names, y_train, y_test = train_test_split(
            names, labels, test_size=test_size,
            random_state=random_state)
        return train_names, test_names

    def _get_data_by_name(self, video_name, data_dir):
        return np.load(f"{data_dir}/{video_name}.npy", allow_pickle=True)

    def _get_data_instance(self, video_name, labels, data_dir, is_augmented=True, is_kinect=True):
        if is_augmented:
            data = []
            labels = []
            audmented_names = [p.stem for p in Path(f"{data_dir}").iterdir() if video_name == p.stem.split("_")[0]]
            for n in audmented_names:
                if is_kinect:
                    d = self._get_data_by_name(video_name + "_kinect", data_dir)
                else:
                    d = self._get_data_by_name(video_name, data_dir)
                data.append(d[0])
                labels.append(d[1])
            data = np.array(data)
            labels = np.array(labels)
        else:
            if is_kinect:
                d = self._get_data_by_name(video_name + "_kinect", data_dir)
            else:
                d = self._get_data_by_name(video_name, data_dir)
            data = np.array([d[0]])
            labels = d[1]
        return data, labels
