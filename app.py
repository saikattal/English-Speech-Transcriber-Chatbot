import streamlit as st
import torch
import librosa
import numpy as np
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from io import BytesIO
import os

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

def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        with st.spinner("Listening..."):
            audio = recognizer.listen(source)
            st.write("Recording complete")
            audio_data = BytesIO(audio.get_wav_data())
            return audio_data

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

st.title("Voice to Text using Wav2Vec2")

if st.button("Record and Transcribe"):
    audio_data = voice_input()
    if audio_data:
        transcription = transcribe_audio(audio_data)
        st.write("Transcription:", transcription)
    else:
        st.write("No audio recorded.")
