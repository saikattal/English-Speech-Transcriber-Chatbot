import streamlit as st
import torch
import librosa
import numpy as np
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from io import BytesIO
import os
from src.helper import voice_input,llm_model_object

HF_TOKEN=os.getenv("HF_TOKEN")
# Constants
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID,token=HF_TOKEN)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID,token=HF_TOKEN)
    return processor, model

# Load model and processor
processor, model = load_model()

# Function to capture and process audio

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



def main():
    st.title("English Voice Enabled Chatbot 👾")

    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    st.subheader("Hi! I am your AI assistant chatbot!😎")

    with st.form(key='voice_input_form'):
        if st.form_submit_button("🎙️ Click to ask me anything!"):
            with st.spinner("Listening..."):
                audio = voice_input()
                text = transcribe_audio(audio)
                response = llm_model_object(text)

                # Update conversation history
                st.session_state.conversation.append({
                    "user": text,
                    "response": response
                })

    st.subheader("💬 Conversation History")
    for entry in st.session_state.conversation:
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**Chatbot:** {entry['response']}")
           
if __name__=='__main__':
    main()
