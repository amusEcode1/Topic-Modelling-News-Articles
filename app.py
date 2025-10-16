import streamlit as st
import joblib

# Load models
@st.cache_resource
def load_models():
    lda_model = joblib.load(lda_pipeline.joblib')
    nmf_model = joblib.load(nmf_pipeline.joblib')
    return lda_model, nmf_model

lda_pipeline, nmf_pipeline = load_models()

# Title
st.title("🧾 News Topic Modelling App")

# Sidebar
model_choice = st.sidebar.selectbox(
    "Choose Topic Modelling Algorithm:",
    ("LDA", "NMF")
)

# User input
user_text = st.text_area("Enter a News Article Text:")

if st.button("Predict Topic"):
    if user_text.strip():
        if model_choice == "LDA":
            model = lda_pipeline
        else:
            model = nmf_pipeline

        # Transform the text and get topic probabilities
        topic_distribution = model.transform([user_text])
        topic_index = topic_distribution.argmax()

        st.success(f"Predicted Topic: Topic {topic_index + 1}")

        # Show topic probabilities
        st.write("Topic probabilities:", topic_distribution)
    else:
        st.warning("Please enter some text.")
