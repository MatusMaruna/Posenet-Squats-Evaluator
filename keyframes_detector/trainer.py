from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.metrics import classification_report

from keyframes_detector.data_creator import KeyFramesDetectorDataCreator
from keyframes_detector.model import KeyFramesDetectorModel


class KeyFramesDetectorTrainer:
    def __init__(self, config):
        self._config = config
        self._data_creator = KeyFramesDetectorDataCreator(config)
        self._session = None
        self._x_train, self._x_test, self._y_train, self._y_test, \
            self._class_weights = self._data_creator.get_feature_datasets()
        self._validation_portion = self._config.dataset.validation_portion / (1.0 - self._config.dataset.test_portion)

        self._ml_model = KeyFramesDetectorModel(self._config)
        self._ml_model.compile()

    def train(self):
        with tf.Session() as self._session:
            self._session.run(tf.global_variables_initializer())
            try:
                self._ml_model.load(self._config.model.path)
            except OSError:
                print("Can't find model. Training from scratch.")
            print('Starting training')
            tensorboard_cb = TensorBoard(log_dir=self._config.training.log_path, histogram_freq=0,
                                         write_graph=True, write_images=True)

            self._ml_model.model.fit(
                self._x_train, self._y_train, epochs=self._config.training.epoch_num,
                validation_split=self._config.dataset.validation_portion, verbose=2,
                batch_size=self._config.training.batch_size, class_weight=self._class_weights,
                callbacks=[ModelCheckpoint(self._config.model.path, 'val_loss', save_best_only=True,
                                           save_weights_only=True),
                           EarlyStopping(monitor='val_auc', patience=self._config.training.lr_decrease_patience,
                                         mode='max', restore_best_weights=True),
                           tensorboard_cb])
            test_loss = self._ml_model.model.evaluate(self._x_test, self._y_test,
                                                      batch_size=self._config.training.batch_size)
            print(f'Test loss: {test_loss}')

            Y_test = np.argmax(self._y_test, axis=1)  # Convert one-hot to index
            y_pred = self._ml_model.predict(self._x_test)
            report = classification_report(Y_test, y_pred)
            print(report)


if __name__ == '__main__':
    from config import general_config

    trainer = KeyFramesDetectorTrainer(general_config.keyframes_detector_pipeline)
    trainer.train()
