from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

from posenet2kinect.data_creator import Pose2KinectDataCreator
from posenet2kinect.model import Pose2KinectModel


class Pose2KinectTrainer:
    def __init__(self, config):
        self._config = config
        self._data_creator = Pose2KinectDataCreator(config)
        self._session = None
        self._x_train, self._x_test, self._y_train, self._y_test = self._data_creator.get_feature_datasets()
        self._validation_portion = self._config.dataset.validation_portion / (1.0 - self._config.dataset.test_portion)

        self._ml_model = Pose2KinectModel(self._config)
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
                batch_size=self._config.training.batch_size,
                callbacks=[ModelCheckpoint(self._config.model.path, 'val_loss', save_best_only=True,
                                           save_weights_only=True),
                           EarlyStopping(monitor='val_loss', patience=self._config.training.lr_decrease_patience),
                           tensorboard_cb])
            test_loss = self._ml_model.model.evaluate(self._x_test, self._y_test,
                                                      batch_size=self._config.training.batch_size)
            print(f'Test loss: {test_loss}')


if __name__ == '__main__':
    from config import general_config

    trainer = Pose2KinectTrainer(general_config.posenet2kinect_pipeline)
    trainer.train()
