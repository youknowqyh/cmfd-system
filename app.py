import os  
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template

from datetime import datetime
import uuid
import imageio
import base64

import tensorflow as tf
from cmfd.utils import *
from cmfd.core import create_cmfd_testing_model

global graph
graph = tf.get_default_graph()

model = create_cmfd_testing_model( 'cmfd/models/pretrained.hd5' )
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    print("here")
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image'].split(',')[1]
    print("-----------Received image--------")
    
    img_name = str(uuid.uuid1())
    with open(f"uploads/inp/{img_name}.png", "wb") as fh:
        fh.write(base64.b64decode(encoded))
        
    img = cv2.imread(f'uploads/inp/{img_name}.png')
    # bgr -> rgb
    img = img[:,:,::-1]
    t0 = datetime.now()
    with graph.as_default():
        pred = cmfd_decoder( model, img )
    t1 = datetime.now()
    print(t1, t0, t1-t0, (t1-t0).total_seconds())
    t = (t1-t0).total_seconds()
    
    imageio.imwrite(f'uploads/out/{img_name}.png', pred)
    with open(f"uploads/out/{img_name}.png", "rb") as image_file:
        out = base64.b64encode(image_file.read())
        
    response = {
        'mask': out.decode('utf-8'),
        'time': t,
    }
        
    return jsonify(response)


if __name__ == '__main__':
    app.run()