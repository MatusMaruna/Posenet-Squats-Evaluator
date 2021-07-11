from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from exercise_quality_recognizer.data_creator import ExerciseQualityDataCreator
from exercise_quality_recognizer.model import ExerciseQualityModel


class ExerciseQualityTrainer:
    def __init__(self, config):
        self._config = config
        self._data_creator = ExerciseQualityDataCreator(config)
        self._session = None
        self._x_train, self._x_test, self._y_train, \
            self._y_test, self._x_val, self._y_val = self._data_creator.get_feature_datasets()
        self._ml_model = ExerciseQualityModel(self._config)
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
                self._x_train, self._y_train, validation_data=(self._x_val, self._y_val),
                epochs=self._config.training.epoch_num, verbose=2,
                batch_size=self._config.training.batch_size,
                callbacks=[ModelCheckpoint(self._config.model.path, 'val_auc', save_best_only=True,
                                           save_weights_only=True),
                           EarlyStopping(monitor='val_auc', verbose=0,
                                         patience=self._config.training.lr_decrease_patience,
                                         mode='max', restore_best_weights=True),
                           tensorboard_cb])
            test_loss = self._ml_model.model.evaluate(self._x_test, self._y_test,
                                                      batch_size=self._config.training.batch_size)

            print(f'Test loss: {test_loss}')


if __name__ == '__main__':
    from config import general_config

    trainer = ExerciseQualityTrainer(general_config.exercise_quality_pipeline)
    trainer.train()
