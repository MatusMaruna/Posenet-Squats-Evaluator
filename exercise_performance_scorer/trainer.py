from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from exercise_performance_scorer.data_creator import ExercisePerformanceDataCreator
from exercise_performance_scorer.model import ExercisePerformanceModel


class ExercisePerformanceTrainer:
    def __init__(self, config):
        self._config = config
        self._data_creator = ExercisePerformanceDataCreator(config)
        self._session = None
        self._train_features, self._train_labels, self._test_features, \
        self._test_labels, self._val_features, self._val_labels = self._data_creator.get_feature_datasets()
        self._ml_model = ExercisePerformanceModel(self._config)
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
                self._train_features, self._train_labels, validation_data=(self._val_features, self._val_labels),
                epochs=self._config.training.epoch_num,verbose=1,
                batch_size=self._config.training.batch_size,
                callbacks=[ModelCheckpoint(self._config.model.path, 'val_loss', save_best_only=True,
                                           save_weights_only=True),
                           EarlyStopping(monitor='val_loss', verbose=0,
                                         patience=self._config.training.lr_decrease_patience,
                                         restore_best_weights=True),
                           tensorboard_cb])
            test_loss = self._ml_model.model.evaluate(self._test_features, self._test_labels,
                                                      batch_size=self._config.training.batch_size)

            print(f'Test loss: {test_loss}')


if __name__ == '__main__':
    from config import general_config

    trainer = ExercisePerformanceTrainer(general_config.exercise_performance_pipeline)
    trainer.train()
