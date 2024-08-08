import streamlit as st
import torch
import librosa
import numpy as np
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from io import BytesIO
import os
from src.helper import voice_input,llm_model_object
#from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
#from datasets import load_dataset

HF_TOKEN=os.getenv("HF_TOKEN")
# Constants
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID,token=HF_TOKEN)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID,token=HF_TOKEN)
    return processor, model

# Set up the page configuration
st.set_page_config(
    page_title="Voice-enabled chatbot",
    #page_icon="🎵",  # You can also use a local path to an image or an emoji
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS for background image and other styles
background_image_url = "https://i.imgur.com/yTo9aCk.png"  # Direct URL to the background image on Imgur

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


st.title("English Voice Enabled Chatbot 👾")
# Load model and processor
processor, model = load_model()

# Function to process audio and generate text
def transcribe_audio(audio_data):
    # Read the audio data from BytesIO stream
    audio_data.seek(0)
    speech_array, sampling_rate = librosa.load(audio_data, sr=16000)
    speech_array = np.array(speech_array, dtype=np.float32)
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Function to convert text to speech
def text_to_speech(text):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    inputs = processor(text=text, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Ensure the input_ids and speaker_embeddings are correctly aligned
    input_ids = inputs["input_ids"]
    if input_ids.size(1) < speaker_embeddings.size(1):
        # Pad input_ids
        padding = speaker_embeddings.size(1) - input_ids.size(1)
        input_ids = torch.nn.functional.pad(input_ids, (0, padding))
    elif input_ids.size(1) > speaker_embeddings.size(1):
        # Truncate input_ids
        input_ids = input_ids[:, :speaker_embeddings.size(1)]

    speech = model.generate_speech(input_ids, speaker_embeddings, vocoder=vocoder)
    return speech.numpy()   



def main():
    

    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    st.subheader("Hi! I am your AI assistant chatbot!😎")

    with st.form(key='voice_input_form'):
        if st.form_submit_button("🎙️ Click to ask me anything!"):
            with st.spinner("Listening..."):
                audio = voice_input()
                text = transcribe_audio(audio)
                response = llm_model_object(text)
                st.write(response)
            #with st.spinner("Processing Voice output..."):
                #speech=text_to_speech(response)
                #st.audio(speech, format="audio/wav",sample_rate=16000)
                
                

                # Update conversation history
                st.session_state.conversation.append({
                    "user": text,
                    "response": response
                })
                

    st.subheader("💬 Conversation History")
    with st.expander("Expand to see the conversation history", expanded=False):
        for entry in st.session_state.conversation:
            st.markdown(f"**You:** {entry['user']}")
            st.markdown(f"**Chatbot:** {entry['response']}")

    
           
if __name__=='__main__':
    main()
