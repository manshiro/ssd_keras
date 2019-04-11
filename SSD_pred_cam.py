import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import math
from PIL import Image, ImageDraw, ImageFont

from ssd import SSD300
from ssd_utils import BBoxUtility

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

voc_classes = ['face', 'hand']
NUM_CLASSES = len(voc_classes) + 1

input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

cam = cv2.VideoCapture(0)

count_max = 1

if __name__ == '__main__':
    count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            print('error')
            break
        # cv2.imshow('keras-pi inspector', capture)
        key = cv2.waitKey(1)
        if key == 27: # when ESC key is pressed break
            break

        count += 1
        if count == count_max:
            inputs = []
            images = []
            images.append(img)
            img_org = img
            img = cv2.resize(img, (300, 300))
            img = image.img_to_array(img)
            inputs.append(img.copy())
            inputs = preprocess_input(np.array(inputs))

            preds = model.predict(inputs, batch_size=1, verbose=1)
            results = bbox_util.detection_out(preds)

            if not np.any(results):
                print('results is empty')
            else:
                for i, img in enumerate(images):
                    # Parse the outputs.
                    det_label = results[i][:, 0]
                    det_conf = results[i][:, 1]
                    det_xmin = results[i][:, 2]
                    det_ymin = results[i][:, 3]
                    det_xmax = results[i][:, 4]
                    det_ymax = results[i][:, 5]

                    # Get detections with confidence higher than 0.6.
                    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]

                    top_conf = det_conf[top_indices]
                    top_label_indices = det_label[top_indices].tolist()
                    top_xmin = det_xmin[top_indices]
                    top_ymin = det_ymin[top_indices]
                    top_xmax = det_xmax[top_indices]
                    top_ymax = det_ymax[top_indices]

                    for i in range(top_conf.shape[0]):
                        xmin = int(round(top_xmin[i] * img.shape[1]))
                        ymin = int(round(top_ymin[i] * img.shape[0]))
                        xmax = int(round(top_xmax[i] * img.shape[1]))
                        ymax = int(round(top_ymax[i] * img.shape[0]))

                        score = top_conf[i]
                        label = int(top_label_indices[i])
                        label_name = voc_classes[label - 1]
                        cv2.rectangle(img_org, (xmin, ymin), (xmax+1, ymax+1), (0, 0, 255), 1)  # red box

                    # Add text
                    text = 'Number: ' + str(top_conf.shape[0])
                    cv2.putText(img_org, text, (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)

            cv2.imshow('detection result', img_org)
            count = 0

    cam.release()
    cv2.destroyAllWindows()
