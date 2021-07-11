from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.metrics import classification_report

from frame_quality_recognizer.data_creator import FrameQualityRecognizerCreator
from frame_quality_recognizer.model import FrameQualityRecognizerModel


class FrameQualityRecognizerTrainer:
    def __init__(self, config):
        self._config = config
        self._data_creator = FrameQualityRecognizerCreator(config)
        self._session = None
        self._x_train, self._y_train,  self._x_valid, self._y_valid, \
            self._x_test, self._y_test = self._data_creator.get_feature_datasets()

        self._ml_model = FrameQualityRecognizerModel(self._config)
        self._ml_model.compile()

    def train(self):
        with tf.Session() as self._session:
            self._session.run(tf.global_variables_initializer())
            try:
                self._ml_model.load(self._config.model.path)
            except Exception:
                print("Can't find model. Training from scratch.")
            print('Starting training')
            tensorboard_cb = TensorBoard(log_dir=self._config.training.log_path, histogram_freq=0,
                                         write_graph=True, write_images=True)

            self._ml_model.model.fit(
                self._x_train, self._y_train, epochs=self._config.training.epoch_num,
                validation_data=(self._x_valid, self._y_valid), verbose=2,
                batch_size=self._config.training.batch_size,
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
            report = classification_report(Y_test, np.argmax(y_pred, axis=1))
            print(report)


if __name__ == '__main__':
    from config import general_config

    trainer = FrameQualityRecognizerTrainer(general_config.frame_quality_recognizer_pipeline)
    trainer.train()
