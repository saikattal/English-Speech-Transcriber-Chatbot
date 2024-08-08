import speech_recognition as sr
import streamlit as st
import os
import google.generativeai as genai
from io import BytesIO
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from langchain_google_genai import ChatGoogleGenerativeAI

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
    

# setup the gemini pro
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

from langchain_core.prompts.prompt import PromptTemplate

template = """The following is a friendly conversation between a human and an AI.
 The AI is talkative and witty and provides lots of specific details from its context.
 If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
def llm_model_object(user_text):
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=False,
        memory=ConversationBufferMemory()
    )
    result = conversation.predict(input=user_text)   
    return result