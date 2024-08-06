import speech_recognition as sr
import streamlit as st
import os
import google.generativeai as genai
from io import BytesIO

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

# Function to capture and process audio
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        #st.write("Recording complete")
        audio_data = BytesIO(audio.get_wav_data())
        return audio_data
    

    
def llm_model_object(user_text):
    #model = "models/gemini-pro"
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    response=model.generate_content(user_text)
    
    result=response.text
    
    return result