import streamlit as st
from breast_cancer.breast_cancer_detection import ml_breast_cancer_model
from cataract_detection.cataract_detection import teachable_machine_model, machine_learning_model


def main():
    # Desabilita el FileUploaderEncodingWarning This change will go in effect after August 15, 2020.
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("AI MODELS")

    activities = ["Breast Cancer Detection", "Cataratas Neural Networks Detection", "Cataratas Machine Learning Detection"]
    choice = st.sidebar.selectbox("Select a model to use", activities)

    if choice == "Breast Cancer Detection":
        ml_breast_cancer_model()
    elif choice == "Cataratas Neural Networks Detection":
        teachable_machine_model()
    elif choice == "Cataratas Machine Learning Detection":
        machine_learning_model()


if __name__ == "__main__":
    main()