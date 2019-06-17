# w251_hw7

Thomas Drage <draget@berkeley.edu>

Cloud storage output:

http://s3.au-syd.cloud-object-storage.appdomain.cloud/hw7draget

Facial SSD NN scripts:

- detect7.py - Run in Ubuntu container on TX2, with CUDA support and Tensorflow. Captures images and uses TensorFlow to detector faces and return bounding boxes. A diagnostic plot is saved locally, whilst each individual face crop is pushed to the MQTT broker for sending to the cloud.
- benchmark7.py - A script for benchmarking the framerate performance of the TensorFlow detector using a single image and the TensorRT optimised network.
- prepare_model.py - A script which takes the publically available frozen model, applies the TensorRT optimisation and then saves back to file.


Scripts per HW3 - see https://github.com/draget/w251_hw3

- forward.py - Run in Alpine container on TX2
- sav.py - Run in Ubuntu container on cloud VM

In addition an Alpine container with MQTT broker (Mosquitto) was present on both TX2 and cloud VM (host name "mosquitto" on a Docker bridge network). On the cloud VM, 1883 from the VM is mapped to the container to accept incoming connections.

- Describe your solution in detail. What neural network did you use? What dataset was it trained on? What accuracy does it achieve?

I used a pre-trained Mobilenet-SSD neural network, available from https://github.com/yeephycho/tensorflow-face-detection, which is a single-shot multibox detector optimised for devices with less compute power available. The Mobilenet architecture is originally developed by Google to augment their heavier Inception model. This example was trained on the WIDERFACE dataset which is available here: http://shuoyang1213.me/WIDERFACE. The accuracy measured as mean averaged precision (see reference below) is close to 80%.

- Does it achieve reasonable accuracy in your empirical tests? Would you use this solution to develop a robust, production-grade system?

Yes - the accuracy is very good, with no false detections. It can detect faces side on as well as front but doesn't deal quite as well with blurring in low light as the simple OpenCV detector.

- What framerate does this method achieve on the Jetson? Where is the bottleneck?

The inference time per benchmarking script is 0.082 seconds. When run in the context of the detect application, speeds around 0.1s, up to 0.15s per frame are observed for inference. This equates to a 8-12 fps range. However, it was observed that time is wasted creating the diagnostic plot (0.1s) and then more saving it to file (0.3s), giving a total frame rate of about 2fps. However - in a production system these are unnecessary and could be disabled if a greater framerate was desired. The rest of the processing is fairly quick, making 8fps, including publishing a reasonable proposition. An example of time distribution for each frame is below:

```
1560701945.6638048  Capture image
Cap Ret True
1560701945.6684606  Run Prediction
1560701945.7848501  Create plot and upload
1560701945.8906243  Crop and encode
1560701945.8920028  Publish
1560701945.8928337  Face complete
1560701945.8938186  Face checking complete
1560701946.2034209  Save complete
1  faces detected.
```

- Which is a better quality detector: the OpenCV or yours?

The Mobilenet-SSD based detector is vastly superior in terms of accuracy. In empirical testing I note significantly less false detections and better quality bounding box definition. Comparative testing results available online show that the accuracy is better than twice as good. Depending on the relative availability of CPU to GPU power, either could be faster, with GPU acceleration able to make the Tensorflow solution a better performer in framerate too. The OpenCV model uses slightly less RAM, however on balance, with a GPU resource available I would certainly choose the neural network.

Reference: https://medium.com/nodeflux/performance-showdown-of-publicly-available-face-detection-model-7c725747094a


