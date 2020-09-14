# FACE RECOGNITION PROJECT ##

## TOOLS : 
- OpenCv
- Imutils
- TensorFlow
- Argparse

## USE : 

- python3 main.py -c [OPTIONS]

### OPTIONS : 
- [1] : to run image encoding.
- [2] : to train the classification model.
- [3] : to run the streaming with face recognition & detection

## PROJECT TREE :
<pre>
.
├── dataset
│   ├── karim
│   │   ├── 20180824.jpg
│   │   ├── ...
│   │   └── IMG.jpg
│   └── unknown
│       ├── 20909684955.jpg
│       ├── ...
│       └── scre.jpg
├── detector
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── embeddings
├── persons
├── main.py
├── modelNN
│   ├── assets
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── openface_nn4.small2.v1.t7
<pre>






