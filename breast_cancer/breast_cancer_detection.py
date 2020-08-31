import streamlit as st
import pandas as pd
import os
from PIL import Image
from pycaret.classification import load_model, predict_model
from extract_features import various_features

width = 700

ml_model = load_model('/mnt/napster_disk/ai_projects/demos/breast_cancer/breast_cancer_model')


def ml_breast_cancer_model():
    st.subheader("Classification using the machine learning model")
    uploaded_file = st.file_uploader("Insert image to analizer", type=None)

    if uploaded_file is not None:
        image_ml = Image.open(uploaded_file)
        st.image(image_ml, caption="The image has been uploaded successfully...", width=width)

        # Send to extract image caracters in dataset
        image_processing = various_features(image_ml)
        df_image_processing = pd.DataFrame(image_processing)
        predictions = predict_model(ml_model, df_image_processing)
        os.remove("/mnt/napster_disk/ai_projects/demos/breast_cancer/analizer.png")
        result = int(predictions['Label'])
        score = float(predictions['Score'])

        if result == 0:
            st.dataframe(predictions.style.highlight_max(axis=0))
            st.success("Bening with {0}".format(score))
        elif result == 1:
            st.dataframe(predictions.style.highlight_max(axis=0))
            st.success("Malign with {0}".format(score))
        elif result == 2:
            st.dataframe(predictions.style.highlight_max(axis=0))
            st.success("Normal with {0}".format(score))
