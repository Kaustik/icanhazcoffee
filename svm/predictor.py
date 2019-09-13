import cv2
import warnings
import os
import numpy as np
import random
from imutils import build_montages
from imutils import paths

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import sys
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    import tensorflow as tf
    from keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing.image import img_to_array
    sys.stderr = stderr

CATEGORIES = ["coffee", "no_coffee"]

IMG_SIZE = 60

def prepare(file):
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("CNN.model")

# grab all image paths in the input directory and randomly sample them
imagePaths = list(paths.list_images("../_images"))
random.shuffle(imagePaths)
imagePaths = imagePaths[:16]
 
# initialize our list of results
results = []

for p in imagePaths:
    print(p)
    orig = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    # pre-process our image by converting it from BGR to RGB channel
	# ordering (since our Keras mdoel was trained on RGB ordering),
	# resize it to 64x64 pixels, and then scale the pixel intensities
	# to the range [0, 1]
    #image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    #f = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
    #image = f.reshape(-1, 75, 75, 1)
    image = orig.astype("float") / 255.0
    #print(image)

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    #print(image)

    # make predictions on the input image
    pred = model.predict(image)
    pred = pred.argmax(axis=1)[0]
    print(pred)
    # an index of zero is the 'parasitized' label while an index of
    # one is the 'uninfected' label
    label = "No Coffee" if pred == 0 else "Coffee!"
    color = (0, 0, 255) if pred == 0 else (0, 255, 0)

    orig = cv2.resize(orig, (128, 128))
    cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        color, 2)

    #print(orig)
    cv2.imwrite('_predictions/'+os.path.basename(p), orig)

    # add the output image to our list of results
    #results.append(orig)

# create a montage using 128x128 "tiles" with 4 rows and 4 columns
#montage = build_montages(results, (128, 128), (4, 4))[0]
 
# show the output montage
#cv2.imshow("Results", montage)
#cv2.waitKey(0)

#prediction = model.predict([prepare(image)], 32, 1)
#print(prediction)
#prediction = list(prediction[0])

#if (CATEGORIES[prediction.index(max(prediction))] == 'coffee'):
#    print('Det finns kaffe!')
#elif (CATEGORIES[prediction.index(max(prediction))] == 'no_coffee'):
#    print('Jag tror tyvärr inte det finns något kaffe.')