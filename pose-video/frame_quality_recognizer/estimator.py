import cv2
import numpy as np
import pickle
import tensorflow as tf

from frame_quality_recognizer.model import FrameQualityRecognizerModel


class FrameQualityRecognizerEstimator:
    def __init__(self, frame_detector_config, model_path=None):
        self.frames_to_collect = 10
        self._sample_size = (224, 224)

        self._config = frame_detector_config
        self._model_path = model_path or self._config.model.path
        self._ml_model = FrameQualityRecognizerModel(self._config)
        self._ml_model.compile()
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())
        self._ml_model.load(self._model_path)
        self._feature_processor = self._load_feature_processor()
        self._label_processor = self._load_label_processor()

    def predict(self, frames, kinect3d_features):
        frames = [cv2.resize(f, self._sample_size, interpolation=cv2.INTER_AREA) for f in frames]
        frames = np.array([frames])
        frames = frames / 255.0
        kinect3d_features = self._feature_processor.transform(kinect3d_features)

        input_features = [frames, np.array([kinect3d_features])]
        predictions = self._ml_model.predict(input_features)
        predictions = self._label_processor.inverse_transform(predictions)[0]
        score = int(predictions[0])
        confidence = round(predictions[1], 2)
        return score, confidence

    def _load_label_processor(self):
        with open(self._config.model.label_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor

    def _load_feature_processor(self):
        with open(self._config.model.feature_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor
