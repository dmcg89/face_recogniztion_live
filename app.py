# Make a flask API for our DL Model


from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
from keras.models import model_from_json
import tensorflow as tf
# import pyrebase
import os
import base64
from io import BytesIO
from keras.models import Model, Sequential
from keras.layers import Input
# from model import create_base_network
from matplotlib.image import imread

from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import os
from keras.callbacks import ModelCheckpoint
import keras
# firebase setup
def create_base_network(in_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Conv2D(128,(7,7),padding='same',input_shape=(in_dims[0],in_dims[1],in_dims[2],),activation='relu',name='conv1'))
    model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool1'))
    model.add(Conv2D(256,(5,5),padding='same',activation='relu',name='conv2'))
    model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(4,name='embeddings'))
    # model.add(Dense(600))
    
    return model

# config = {
#   "apiKey": os.environ['FIREBASE_API_KEY'],
#   "authDomain": "ds23-mnist.firebaseapp.com",
#   "databaseURL": "https://ds23-mnist.firebaseio.com",
#   "projectId": "ds23-mnist",
#   "storageBucket": "ds23-mnist.appspot.com",
#   "serviceAccount": "firebase-private-key.json",
#   "messagingSenderId": "69895286672"
# }

# firebase = pyrebase.initialize_app(config)
# db = firebase.database()

app = Flask(__name__)
api = Api(app, version='1.0', title='MNIST Classification', description='CNN for Mnist')
ns = api.namespace('Make_School', description='Methods')

single_parser = api.parser()
single_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

model = load_model('my_model.h5')
# graph = tf.get_default_graph()
graph = tf.compat.v1.get_default_graph()

# Model reconstruction from JSON file
# with open('model_architecture.json', 'r') as f:
#     model = model_from_json(f.read())
#
# # Load weights into the new model
# model.load_weights('model_weights.h5')
anchor_input = Input((480, 640, 1), name='anchor_input')
Shared_DNN = create_base_network([480, 640, 1])
encoded_anchor = Shared_DNN(anchor_input)
trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x

@ns.route('/prediction')
class CNNPrediction(Resource):
    """Uploads your data to the CNN"""
    @api.doc(parser=single_parser, description='Upload an mnist image')
    def post(self):
        args = single_parser.parse_args()
        image_file = args.file
        image_file.save('milad.pgm')
        img = Image.open('milad.pgm')
        img = imread(img)
        x = np.squeeze(np.asarray(img))
        x = normalize(x)
        
        x = trained_model.predict(x.reshape(480, 640, 1))
        # image_red = img.resize((28, 28))
        # image = img_to_array(image_red)
        # print(image.shape)
        # x = image.reshape(1, 480, 640, 1)
        # x = x/255

        trained_model.load_weights("weights.hdf5")
        x = trained_model.predict(img.reshape(-1,480, 640, 1))
        # This is not good, because this code implies that the model will be
        # loaded each and every time a new request comes in.
        # model = load_model('my_model.h5')
        with graph.as_default():
            out = model.predict(x)
        print(out[0])
        print(np.argmax(out[0]))
        r = np.argmax(out[0])

        return {'prediction': str(r)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)