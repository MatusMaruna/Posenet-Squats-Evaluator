import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Pose2KinectDataCreator:
    def __init__(self, config):
        self._config = config
        self._num_of_inputs = 26
        self._num_of_outputs = 26

    def get_feature_datasets(self):
        data = pd.DataFrame()
        for csv_posix in Path(self._config.dataset.path).iterdir():
            if csv_posix.suffix != '.csv':
                continue
            d = pd.read_csv(str(csv_posix))
            data = pd.concat([data, d], axis=0)

        # Remove frame, score, z and unnamed columns
        data = data.loc[:, ~data.columns.str.contains('^Unnamed|^Frame|^Pose|Eye|Ear|_z|_score|_Score')]
        # Randomize data
        data = data.sample(frac=1).reset_index(drop=True)
        shuffled_data = data.sample(n=(data.shape[0]), random_state=self._config.dataset.random_state)
        # Split into inputs and labels
        raw_inputs = shuffled_data.iloc[:, 0:self._num_of_inputs]
        raw_outputs = shuffled_data.drop(raw_inputs.columns, axis=1)

        inputs = self._process_inputs(raw_inputs)
        outputs = self._process_outputs(raw_outputs)

        x_train, x_test, y_train, y_test = train_test_split(
            inputs, outputs, test_size=self._config.dataset.test_portion,
            random_state=self._config.dataset.random_state)
        return x_train, x_test, y_train, y_test

    def _process_inputs(self, raw_inputs):
        # Scale data
        scaler = MinMaxScaler()
        inputs = scaler.fit_transform(raw_inputs)
        with open(self._config.model.feature_processor_path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        return inputs

    def _process_outputs(self, raw_outputs):
        # Scale data
        scaler = MinMaxScaler()
        outputs = scaler.fit_transform(raw_outputs)
        with open(self._config.model.label_processor_path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        return outputs

