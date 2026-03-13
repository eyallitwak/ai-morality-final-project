import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Deepfake Detector")

# caching the ML model
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")

classifier = load_model()

st.title("Deepfake Detection Simulator")

uploaded_file = st.file_uploader("Upload a face to test", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([2, 1]) 
    
    # human decision part
    with col1:
        st.markdown("Is this person real?")
        st.image(image, use_container_width=True) 
        
        user_choice = st.radio("Regardless of the AI, what is YOUR judgment?", ["Real", "Fake"], index=None)
        
        if user_choice is not None:
            st.info(f"You locked in: {user_choice}")

    # model decision "AI"
    with col2:
        st.markdown("AI Companion")
        st.write("Need a second opinion? Adjust the strictness and ask the AI.")
        
        threshold = st.slider("AI 'Real' Threshold", 0.0, 1.0, 0.50, 0.05)
        
        if st.button("Run AI Detection"):
            with st.spinner("Thinking..."):
                results = classifier(image)
                
                real_score = 0.0
                for res in results:
                    if 'real' in res['label'].lower():
                        real_score = res['score']
                        break
                
                st.markdown("AI Verdict")
                if real_score >= threshold:
                    st.success("I believe this is a real person")
                else:
                    st.error("I believe this is a deepfake")
                
                # actual real score of the model. not sure if needed to be seen by users
                # st.write(f"Real score: {real_score:.2%}")
                
                # raw model output - debug
                with st.expander("See Raw Model Output"):
                    st.write(results)