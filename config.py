from easydict import EasyDict
from pathlib import Path


general_config = EasyDict()
general_config.root_path = str(Path(__file__).parent)

general_config.service = EasyDict()
general_config.service.host = "0.0.0.0"
general_config.service.port = "5000"


general_config.kinect_pipeline = EasyDict()
general_config.kinect_pipeline.dataset = EasyDict()
general_config.kinect_pipeline.dataset.validation_portion = 0.2
general_config.kinect_pipeline.dataset.test_portion = 0.1
general_config.kinect_pipeline.dataset.path = f"{general_config.root_path}/data/kinect_good_preprocessed"
general_config.kinect_pipeline.dataset.random_state = 10

general_config.kinect_pipeline.model = EasyDict()
general_config.kinect_pipeline.model.path = f"{general_config.root_path}/models/kinect_model.pickle"
general_config.kinect_pipeline.model.feature_processor_path = \
    f"{general_config.root_path}/models/kinect_feature_processor.pickle"
general_config.kinect_pipeline.model.label_processor_path = \
    f"{general_config.root_path}/models/kinect_label_processor.pickle"

general_config.kinect_pipeline.training = EasyDict()
general_config.kinect_pipeline.training.batch_size = 128
general_config.kinect_pipeline.training.epoch_num = 500
general_config.kinect_pipeline.training.lr_decrease_patience = 6
general_config.kinect_pipeline.training.learning_rate = 1e-4
general_config.kinect_pipeline.training.log_path = f"{general_config.root_path}/logs"


general_config.posenet2kinect_pipeline = EasyDict()
general_config.posenet2kinect_pipeline.dataset = EasyDict()
general_config.posenet2kinect_pipeline.dataset.validation_portion = 0.2
general_config.posenet2kinect_pipeline.dataset.test_portion = 0.1
general_config.posenet2kinect_pipeline.dataset.path = f"{general_config.root_path}/data/posenet2kinect_data"
general_config.posenet2kinect_pipeline.dataset.random_state = 10

general_config.posenet2kinect_pipeline.model = EasyDict()
general_config.posenet2kinect_pipeline.model.path = f"{general_config.root_path}/models/posenet2kinect_model.pickle"
general_config.posenet2kinect_pipeline.model.feature_processor_path = \
    f"{general_config.root_path}/models/posenet2kinect_input_processor.pickle"
general_config.posenet2kinect_pipeline.model.label_processor_path = \
    f"{general_config.root_path}/models/posenet2kinect_output_processor.pickle"

general_config.posenet2kinect_pipeline.training = EasyDict()
general_config.posenet2kinect_pipeline.training.batch_size = 128
general_config.posenet2kinect_pipeline.training.epoch_num = 1000
general_config.posenet2kinect_pipeline.training.lr_decrease_patience = 6
general_config.posenet2kinect_pipeline.training.learning_rate = 1e-4
general_config.posenet2kinect_pipeline.training.log_path = f"{general_config.root_path}/logs"


general_config.keyframes_detector_pipeline = EasyDict()
general_config.keyframes_detector_pipeline.dataset = EasyDict()
general_config.keyframes_detector_pipeline.dataset.validation_portion = 0.2
general_config.keyframes_detector_pipeline.dataset.test_portion = 0.3
general_config.keyframes_detector_pipeline.dataset.path = f"{general_config.root_path}/data/keyframes_detector_data"
general_config.keyframes_detector_pipeline.dataset.random_state = 10

general_config.keyframes_detector_pipeline.model = EasyDict()
general_config.keyframes_detector_pipeline.model.path = f"{general_config.root_path}/models/keyframes_detector_model.pickle"
general_config.keyframes_detector_pipeline.model.feature_processor_path = \
    f"{general_config.root_path}/models/keyframes_detector_feature_processor.pickle"
general_config.keyframes_detector_pipeline.model.label_processor_path = \
    f"{general_config.root_path}/models/keyframes_detector_label_processor.pickle"

general_config.keyframes_detector_pipeline.training = EasyDict()
general_config.keyframes_detector_pipeline.training.batch_size = 128
general_config.keyframes_detector_pipeline.training.epoch_num = 10000
general_config.keyframes_detector_pipeline.training.lr_decrease_patience = 15
general_config.keyframes_detector_pipeline.training.learning_rate = 1e-5
general_config.keyframes_detector_pipeline.training.log_path = f"{general_config.root_path}/logs"



general_config.exercise_quality_pipeline = EasyDict()
general_config.exercise_quality_pipeline.dataset = EasyDict()
general_config.exercise_quality_pipeline.dataset.validation_portion = 0.2
general_config.exercise_quality_pipeline.dataset.test_portion = 0.1
general_config.exercise_quality_pipeline.dataset.path = f"{general_config.root_path}/data/exercise_quality_data"
general_config.exercise_quality_pipeline.dataset.random_state = 10

general_config.exercise_quality_pipeline.model = EasyDict()
general_config.exercise_quality_pipeline.model.path = f"{general_config.root_path}/models/exercise_quality_model.pickle"
general_config.exercise_quality_pipeline.model.feature_processor_path = \
    f"{general_config.root_path}/models/exercise_quality_input_processor.pickle"
general_config.exercise_quality_pipeline.model.label_processor_path = \
    f"{general_config.root_path}/models/exercise_quality_output_processor.pickle"

general_config.exercise_quality_pipeline.training = EasyDict()
general_config.exercise_quality_pipeline.training.batch_size = 8
general_config.exercise_quality_pipeline.training.epoch_num = 15000
general_config.exercise_quality_pipeline.training.lr_decrease_patience = 10
general_config.exercise_quality_pipeline.training.learning_rate = 3e-5
general_config.exercise_quality_pipeline.training.log_path = f"{general_config.root_path}/logs"

general_config.exercise_performance_pipeline = EasyDict()
general_config.exercise_performance_pipeline.dataset = EasyDict()
general_config.exercise_performance_pipeline.dataset.validation_portion = 0.2
general_config.exercise_performance_pipeline.dataset.test_portion = 0.1
general_config.exercise_performance_pipeline.dataset.path = f"{general_config.root_path}/data/exercise_performance_data"
general_config.exercise_performance_pipeline.dataset.scores_path = f"{general_config.root_path}/data/scores.csv"
general_config.exercise_performance_pipeline.dataset.random_state = 10

general_config.exercise_performance_pipeline.model = EasyDict()
general_config.exercise_performance_pipeline.model.path = \
    f"{general_config.root_path}/models/exercise_performance_model.pickle"
general_config.exercise_performance_pipeline.model.feature_processor_path = \
    f"{general_config.root_path}/models/exercise_performance_feature_processor.pickle"
general_config.exercise_performance_pipeline.model.label_processor_path = \
    f"{general_config.root_path}/models/exercise_performance_label_processor.pickle"

general_config.exercise_performance_pipeline.training = EasyDict()
general_config.exercise_performance_pipeline.training.batch_size = 16
general_config.exercise_performance_pipeline.training.epoch_num = 500
general_config.exercise_performance_pipeline.training.lr_decrease_patience = 10
general_config.exercise_performance_pipeline.training.learning_rate = 3e-4
general_config.exercise_performance_pipeline.training.log_path = f"{general_config.root_path}/logs"

general_config.frame_quality_recognizer_pipeline = EasyDict()
general_config.frame_quality_recognizer_pipeline.dataset = EasyDict()
general_config.frame_quality_recognizer_pipeline.dataset.validation_portion = 0.2
general_config.frame_quality_recognizer_pipeline.dataset.test_portion = 0.1
general_config.frame_quality_recognizer_pipeline.dataset.path = \
    f"{general_config.root_path}/data/frame_quality_recognizer_data"
general_config.frame_quality_recognizer_pipeline.dataset.random_state = 10

general_config.frame_quality_recognizer_pipeline.model = EasyDict()
general_config.frame_quality_recognizer_pipeline.model.path = \
    f"{general_config.root_path}/models/frame_quality_recognizer_model.pickle"
general_config.frame_quality_recognizer_pipeline.model.feature_processor_path = \
    f"{general_config.root_path}/models/frame_quality_recognizer_feature_processor.pickle"
general_config.frame_quality_recognizer_pipeline.model.label_processor_path = \
    f"{general_config.root_path}/models/frame_quality_recognizer_label_processor.pickle"

general_config.frame_quality_recognizer_pipeline.training = EasyDict()
general_config.frame_quality_recognizer_pipeline.training.batch_size = 8
general_config.frame_quality_recognizer_pipeline.training.epoch_num = 3000
general_config.frame_quality_recognizer_pipeline.training.lr_decrease_patience = 3
general_config.frame_quality_recognizer_pipeline.training.learning_rate = 3e-4
general_config.frame_quality_recognizer_pipeline.training.log_path = f"{general_config.root_path}/logs"
