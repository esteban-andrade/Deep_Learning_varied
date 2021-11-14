"""
AUTHOR: ESTEBAN ANDRADE

Suggestions:
1. Increase dataset for training
2. Add better labeling and more consistent labels
3. Performance tune Output Model in order to increase speed and accuracy.
3. Choose different pretrained model
4. Increase number of epochs for learning 
5. If we want 3D reconstruction we will need the Pose of the Camera in order to use the transforms of In order to get the Position of the vehicles with respect to the Bird 
eye camera

"""

import cv2
import uuid
import os
import time
import sys
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import numpy as np
from matplotlib import pyplot as plt


FINAL_CONFIG = os.path.join(
    'model', 'workspace', 'models', 'my_ssd_mobnet', 'export', 'pipeline.config')
FINAL_CHECK_POINT = os.path.join(
    'model', 'workspace', 'models', 'my_ssd_mobnet', 'export', 'checkpoint')

LABELMAP = os.path.join('model', 'workspace', 'annotations', 'label_map.pbtxt')


def LoadPipeline(FINAL_CONFIG):
    configs = config_util.get_configs_from_pipeline_file(
        FINAL_CONFIG)

    return configs


def BuildDetectionModel(CONFIG):
    detection_model = model_builder.build(
        model_config=CONFIG['model'], is_training=False)

    return detection_model


def restoreCheckPoint(MODEL, FINAL_CHECK_POINT_PATH):
    ckpt = tf.compat.v2.train.Checkpoint(model=MODEL)
    ckpt.restore(os.path.join(FINAL_CHECK_POINT_PATH, 'ckpt-0')
                 ).expect_partial()

    return ckpt


def getCategoryIndices(LABELMAP):
    category_index = label_map_util.create_category_index_from_labelmap(
        LABELMAP)
    return category_index


@tf.function
def detect_fn(image, model):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections


def main():
    configs = LoadPipeline(FINAL_CONFIG)
    detection_model = BuildDetectionModel(configs)
    ckpt = restoreCheckPoint(detection_model, FINAL_CHECK_POINT)
    category_index = getCategoryIndices(LABELMAP)
    capture = cv2.VideoCapture("video_01.mp4")
    framerate = capture.get(60)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time.sleep(3)
    while capture.isOpened():

        sucess, frame = capture.read()
        if sucess:
            image_np = np.array(frame)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            input_tensor = tf.convert_to_tensor(
                image_np_expanded, dtype=tf.float32)
            detections = detect_fn(input_tensor, detection_model)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(
                np.int64)
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
            # cv2.imshow('Car Detector', image_np_with_detections)
            cv2.imshow('Car Detector',  cv2.resize(
                image_np_with_detections, (1280, 720)))

        else:
            print('no video')
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
