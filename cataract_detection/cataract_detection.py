import streamlit as st
import pandas as pd
import os
import numpy as np
import tensorflow
from pycaret.classification import load_model, predict_model
from PIL import Image, ImageOps
from extract_features import various_features

threshold = 0.95
width = 700

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model_teachable = tensorflow.keras.models.load_model('/mnt/napster_disk/ai_projects/ai_models/cataract_detection'
                                                     '/keras_model1.h5')
model_ml = load_model('/mnt/napster_disk/ai_projects/ai_models/cataract_detection/pycaret_model')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


@st.cache(allow_output_mutation=True)
def clasify(image):
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    if image_array.shape == (224, 224, 3):
        # display the resized image
        # image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model_teachable.predict(data)
        return prediction
    else:
        pass


def teachable_machine_model():
    st.subheader("Validation with model generated with Neural Networks")
    uploaded_file = st.file_uploader("Enter an image to analyze", type='png')

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="The file has been uploaded successfully...", width=width)
        pred = clasify(image)

        if pred is not None:
            # Cataract
            if pred[0][0] >= threshold:
                st.success("Catarata with probability: {}".format(str(pred[0][0])))
            # Normal
            elif pred[0][1] >= threshold:
                st.success("Normal with probability: {}".format(str(pred[0][1])))
            else:
                st.warning("The probability is not reliable in the validation of the image with the following "
                           "ratings"
                           "Cataract: {0} \n"
                           "Normal: {1}".format(pred[0][0], pred[0][1]))
        else:
            st.warning("ERROR image shape is not support")

        image.close()


def machine_learning_model():
    st.subheader("Validation with model generated with machine learning")
    uploaded_file = st.file_uploader("Enter an image to analyze", type=None)

    if uploaded_file is not None:
        image_ml = Image.open(uploaded_file)
        st.image(image_ml, caption="The file has been uploaded successfully...", width=width)

        # Send to extract image caracters in dataset
        image_processing = various_features(image_ml)
        df_image_processing = pd.DataFrame(image_processing)
        predictions = predict_model(model_ml, df_image_processing)
        os.remove("/mnt/napster_disk/ai_projects/ai_models/analizer.png")
        result = int(predictions['Label'])
        score = float(predictions['Score'])

        if result == 0:
            st.success("Glaucoma with {0} probability".format(score))
        elif result == 1:
            st.success("Normal with {0} probability".format(score))
        elif result == 2:
            st.success("Issues with retina, {0} probability".format(score))
        elif result == 3:
            st.success("Es catarata with {0} probability".format(score))
