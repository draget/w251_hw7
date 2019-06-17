#!/usr/bin/python3

###
# Face Detector with MQTT
# HW 7 - DNN version
# <draget@berkeley.edu>
###

import time

print(time.time(), " Import Libs");


from PIL import Image
import sys
import os
import urllib
import tensorflow.contrib.tensorrt as trt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
from tf_trt_models.detection import download_detection_model, build_detection_graph

import paho.mqtt.client as mqtt
import cv2


# ### Load the (optimised) frozen graph

print(time.time(), " Load Graph");

output_dir=''
trt_graph = tf.GraphDef()
with open(os.path.join(output_dir, "tensorrt_model.pb"), 'rb') as f:
  trt_graph.ParseFromString(f.read())


# ### A few magical constants
INPUT_NAME='image_tensor'
BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
MASKS_NAME='detection_masks'
NUM_DETECTIONS_NAME='num_detections'

input_names = [INPUT_NAME]
output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

# ### Create session and load graph

print(time.time(), " Create tf sess");


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

print(time.time(), " Import Graph");

tf.import_graph_def(trt_graph, name='')

print(time.time(), " Setup tf vars");

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))


print(time.time(), " MQTT connect");

# Create client and connect to MQTT broker
client = mqtt.Client()
client.on_connect = on_connect

client.connect("mosquitto", 1883, 60)

# ### Load and Preprocess Image

print(time.time(), " Process images");

cap = cv2.VideoCapture(1)

while(True):

    print(time.time(), " Capture image")

    ret, image = cap.read()

    print("Cap Ret", ret)

    image_pil = Image.fromarray(image)

    image_resized = np.array(image_pil.resize((300, 300)))
#    image = np.array(image)

#    print(image_resized)

    # ### Run network on Image

    print(time.time(), " Run Prediction");

    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
        tf_input: image_resized[None, ...]
    })

    boxes = boxes[0] # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = num_detections[0]

    # ### Display Results

    print(time.time(), " Create plot and upload");

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(image)

    # suppress boxes that are below the threshold.. 
    DETECTION_THRESHOLD = 0.5

    valid_detections = 0

    # plot boxes exceeding score threshold
    for i in range(int(num_detections)):
        if scores[i] < DETECTION_THRESHOLD:
            continue

        valid_detections = valid_detections + 1

        # scale box to image coordinates
        box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])

        # display rectangle
        patch = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], color='g', alpha=0.3)
        ax.add_patch(patch)

        # display class index and score
        plt.text(x=box[1] + 10, y=box[2] - 10, s='%d (%0.2f) ' % (classes[i], scores[i]), color='w')

        y = int(box[0])
        x = int(box[1])
        w = int(box[3] - box[1])
        h = int(box[2] - box[0])

        print(time.time(), " Crop and encode");

        # Crop the face
        face_mat =  image[y:y+h, x:x+w]
        # Encode as PNG
        rc, png = cv2.imencode('.png', face_mat)
        msg = png.tobytes()

        print(time.time(), " Publish");

        # Publish to MQTT topic
        client.publish("faces", payload = msg, qos = 0, retain = False)

        print(time.time(), " Face complete");

    print(time.time(), " Face checking complete");

    plt.savefig("det.png")

    plt.close(fig)

    print(time.time(), " Save complete");

    print(valid_detections, " faces detected.")

# ### Close session to release resources


print(time.time(), " Close");

tf_sess.close()

