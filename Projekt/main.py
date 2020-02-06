import os

# import magic
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import cv2 as cv
import glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from pathlib import Path

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


#########################################################


def finger_function():
    g_kernel = cv.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv.CV_32F)
    # Wczytanie danych
    X_data = []
    files = glob.glob('-/Projekt/newDB/Fingerprints/fp_1/*.tif')
    for myFile in files:
        print(myFile)
        image = cv.imread(myFile, 0)
        crop_img = image[120:220, 75:175]
        filtered_img = cv.filter2D(crop_img, cv.CV_8UC3, g_kernel)
        X_data.append(filtered_img)
        #print('X_data shape:', np.array(X_data).shape)

    #print(X_data)

    Xt_data = []
    files = glob.glob('-Projekt/newDBT/Fingerprints/fp_1/*.tif')
    for myFile in files:
        print(myFile)
        image = cv.imread(myFile, 0)
        crop_img = image[120:220, 75:175]
        filtered_img = cv.filter2D(crop_img, cv.CV_8UC3, g_kernel)
        Xt_data.append(filtered_img)
        #print('X_data shape:', np.array(Xt_data).shape)

    #print(Xt_data)

    Xs_data = []
    files = glob.glob('-Projekt/checkphoto/*.tif')
    for myFile in files:
        print(myFile)
        image = cv.imread(myFile, 0)
        crop_img = image[120:220, 75:175]
        filtered_img = cv.filter2D(crop_img, cv.CV_8UC3, g_kernel)
        Xs_data.append(filtered_img)
        #print('X_data shape:', np.array(Xs_data).shape)

    #print(Xs_data)

    Xp_data = []
    files = glob.glob('-Projekt/newDBP/Fingerprints/*.tif')
    for myFile in files:
        print(myFile)
        image = cv.imread(myFile, 0)
        crop_img = image[120:220, 75:175]
        filtered_img = cv.filter2D(crop_img, cv.CV_8UC3, g_kernel)
        Xp_data.append(filtered_img)
        #print('X_data shape:', np.array(Xp_data).shape)

    #print(Xp_data)
    # Dostosowanie formatu tablic
    X_data = np.array(X_data, dtype=np.float32)
    Xt_data = np.array(Xt_data, dtype=np.float32)
    Xs_data = np.array(Xs_data, dtype=np.float32)
    Xp_data = np.array(Xp_data, dtype=np.float32)

    X_data = X_data.astype('float32') / 255.
    Xt_data = Xt_data.astype('float32') / 255.
    Xs_data = Xs_data.astype('float32') / 255.
    Xp_data = Xp_data.astype('float32') / 255.

    X_data = np.reshape(X_data, (len(X_data), 100, 100, 1))
    Xt_data = np.reshape(Xt_data, (len(Xt_data), 100, 100, 1))
    Xs_data = np.reshape(Xs_data, (len(Xs_data), 100, 100, 1))
    Xp_data = np.reshape(Xp_data, (len(Xp_data), 100, 100, 1))

    my_file = Path('-/Projekt/my_model.h5')
    if my_file.exists():
        autoencoder = load_model('my_model.h5')
        print("MODEL WCZYTANY")
    else:
        # Tworzenie modelu
        input_img = Input(shape=(100, 100, 1))
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        # uczenie modelu
        autoencoder.fit(X_data, X_data,
                        epochs=10,
                        batch_size=250,
                        shuffle=True,
                        validation_data=(Xt_data, Xt_data),
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    # przetworzenie danych przez autoencoder
    #encoded_imgs = autoencoder.predict(Xt_data)
    encoded_s = autoencoder.predict(Xs_data)
    encoded_p = autoencoder.predict(Xp_data)
    print("ok")
    from scipy import signal
    k = 0
    s = 1
    m = 10
    for j in range(s):
        for l in range(m):

            img = encoded_s[j].reshape(100, 100)
            img2 = encoded_p[l].reshape(100, 100)
            cor = signal.correlate2d(img, img, mode="valid")
            cor1 = signal.correlate2d(img2, img2, mode="valid")
            if cor == cor1:
                k = k + 1
            else:
                k = k
    return k

    # Zapisanie modelu
    autoencoder.save('my_model.h5')


##########################################################
@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('WYBIERZ ELEMENT')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            z = finger_function()
            if z > 0:
                flash('ISTNIEJE W BAZIE')
            else:
                flash('NIE ISTNIEJE W BAZIE')
            return redirect('/')
        else:
            flash('DOZWOLONE FORMATY TO tif')
            return redirect(request.url)


if __name__ == "__main__":
    app.run()
