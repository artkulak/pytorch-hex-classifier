import os
# import magic
import urllib.request
from app import app

from modelCNN import Net
import torch
import binascii

from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.preprocessing import OneHotEncoder

n_hidden = 128
labels = os.listdir('dataset/')
CHARS = np.array(list(map(chr, list(np.arange(97, 122)))) + list(np.arange(0, 10)))
NUM_CHARS = len(CHARS)
LINE_LENGTH = 20
ohe = OneHotEncoder()
ohe.fit(CHARS.reshape(-1, 1))

CAT_MAPPING = {0: 'bmp', 1: 'flv', 2: 'jpg', 3: 'mp3', 4: 'mp4', 5: 'pdf', 6: 'png', 7: 'wav'}
N_CLASSES = len(CAT_MAPPING)

@app.route('/')
def upload_form():
    '''
    Renders the upload.html file
    :return: The template of the web page
    '''
    return render_template('upload.html')

def to_tensor(line, ohe, maxchars):
    '''
    One Hot Encodes the line of symbols and then transforms to torch tensor, to pass into the model
    :param line: Line of char symbols
    :param ohe: OneHotEncoder instance
    :param maxchars: Number of chars to take from the both sides of line
    :return: Torch Tensor of one hot encoded line
    '''
    return ohe.transform(np.array(list(line[:maxchars].decode('ascii')) + list(line[-maxchars:].decode('ascii'))).reshape(-1, 1)).todense()



def load_model():
    '''
    Loads the pretrained CNN model
    :return: torch model
    '''
    model = Net(N_CLASSES)
    model.load_state_dict(torch.load('model.pt'))

    model.eval()
    return model

model = load_model().double()


def predict(model, file):
    '''
    Predicts the class of the given file
    :param model: torch pretrained model
    :param file: the file of which you would like to predict the format
    :return: The category of the file
    '''
    with open(file, 'rb') as f:
        content = f.read()
        file_array = binascii.hexlify(content)
        inp = torch.from_numpy(to_tensor(file_array, ohe, LINE_LENGTH)).reshape(1, 1, 2*LINE_LENGTH, NUM_CHARS).double()

        output = model(inp)
        return CAT_MAPPING[np.argmax(output.detach().numpy())]

@app.route('/', methods=['POST'])
def upload_file():
    '''
    Method which allows the file upload
    :return: The redirect to the page with the rendered redicted class of the file
    '''
    if request.method == 'POST':

        print(request.files)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label = predict(model, 'TEST/' + filename)


            flash(label)
            return redirect('/')
        else:
            flash('An error happened!')
            return redirect(request.url)


if __name__ == "__main__":
    # starts the flask app
    app.run()