import streamlit as st
from breast_cancer.breast_cancer_detection import ml_breast_cancer_model


def main():
    # Desabilita el FileUploaderEncodingWarning This change will go in effect after August 15, 2020.
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("AI MODELS")

    activities = ["Breast Cancer Detection", "Cataratas Neural Networks Detection", "Cataratas Machine Learning Detection"]
    choice = st.sidebar.selectbox("Select a model to use", activities)

    if choice == "Breast Cancer Detection":
        ml_breast_cancer_model()
    elif choice == "Cataratas Neural Networks Detection":
        pass
    elif choice == "Cataratas Machine Learning Detection":
        pass


if __name__ == "__main__":
    main()