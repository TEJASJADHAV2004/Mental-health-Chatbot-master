import streamlit as st
import nltk
import ssl
import pickle
import numpy as np
import json
import random
from nltk.stem import WordNetLemmatizer

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data
@st.cache_resource
def load_model_data():
    try:
        # Load model (if available)
        from keras.models import load_model
        model = load_model('model.h5')
        
        # Load processed data
        words = pickle.load(open('texts.pkl', 'rb'))
        classes = pickle.load(open('labels.pkl', 'rb'))
        
        # Load intents
        with open('intents.json', 'r') as file:
            intents = json.load(file)
            
        return model, words, classes, intents
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# Load intents only (fallback)
@st.cache_resource
def load_intents_only():
    try:
        with open('intents.json', 'r') as file:
            intents = json.load(file)
        return intents
    except Exception as e:
        st.error(f"Error loading intents: {e}")
        return None

# Clean up sentence
def clean_up_sentence(sentence):
    try:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    except:
        return sentence.lower().split()

# Bag of words
def bow(sentence, words, show_details=True):
    try:
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)
    except:
        return np.array([0] * len(words))

# Predict class
def predict_class(sentence, model, words, classes):
    try:
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.15
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list
    except:
        return []

# Get response
def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                return result
    return "I'm here to help and support you. Could you please tell me more about what you're going through?"

# Rule-based chatbot response
def chatbot_response(msg, model=None, words=None, classes=None, intents=None):
    try:
        # Rule-based responses for better accuracy
        msg_lower = msg.lower()
        
        # Mental health specific responses
        if any(word in msg_lower for word in ['anxiety', 'anxious', 'worried', 'panic']):
            anxiety_responses = [
                "I understand that anxiety can be overwhelming. Can you tell me more about what's making you feel anxious?",
                "Anxiety is very common and treatable. What specific situations or thoughts trigger your anxiety?",
                "I'm here to help with your anxiety. Would you like to talk about what's causing these feelings?",
                "Feeling anxious is difficult. Can you describe what you're experiencing right now?"
            ]
            return random.choice(anxiety_responses)
            
        elif any(word in msg_lower for word in ['depressed', 'depression', 'hopeless', 'worthless']):
            depression_responses = [
                "I'm sorry you're feeling this way. Depression is a serious condition, but you're not alone. Can you tell me more about how you've been feeling?",
                "Thank you for sharing that with me. Depression affects many people. What has been the most difficult part for you?",
                "I want you to know that what you're feeling is valid. Can you help me understand what's been going on?",
                "It takes courage to reach out. How long have you been feeling this way?"
            ]
            return random.choice(depression_responses)
            
        elif any(word in msg_lower for word in ['stressed', 'stress', 'overwhelmed', 'burned out']):
            stress_responses = [
                "Stress can be really challenging to deal with. What's been causing you the most stress lately?",
                "I hear that you're feeling overwhelmed. Can you tell me what's been on your mind?",
                "Stress affects us all differently. What would help you feel more manageable right now?",
                "It sounds like you're going through a tough time. What's been the biggest source of stress for you?"
            ]
            return random.choice(stress_responses)
            
        elif any(word in msg_lower for word in ['lonely', 'alone', 'isolated', 'empty']):
            loneliness_responses = [
                "Feeling lonely can be really painful. I'm here with you right now. Can you tell me more about these feelings?",
                "You're not alone in feeling this way. Many people experience loneliness. What's been making you feel most isolated?",
                "I'm glad you reached out. Loneliness is hard to bear. What would help you feel more connected?",
                "Thank you for sharing this with me. What does loneliness feel like for you?"
            ]
            return random.choice(loneliness_responses)
            
        elif any(word in msg_lower for word in ['suicide', 'kill myself', 'end it all', 'don\'t want to live']):
            crisis_responses = [
                "I'm very concerned about you. Please know that you matter and there is help available. Contact emergency services or call 988 (Suicide & Crisis Lifeline) immediately.",
                "Your life has value and meaning. Please reach out for immediate help: Call 988 or go to your nearest emergency room. I'm here to support you.",
                "I'm worried about you. These feelings can be temporary, but please get immediate help. Call 988 or emergency services right now."
            ]
            return random.choice(crisis_responses)
            
        # Use ML model if available
        if model and words and classes and intents:
            ints = predict_class(msg, model, words, classes)
            res = getResponse(ints, intents)
            return res
        
        # Fallback responses
        fallback_responses = [
            "I'm here to listen and support you. Can you tell me more about what's on your mind?",
            "I want to help you. Could you share more about how you're feeling or what you're going through?",
            "I'm here for you. What would you like to talk about today?",
            "Thank you for reaching out. Can you help me understand what you're experiencing?"
        ]
        return random.choice(fallback_responses)
        
    except Exception as e:
        return "I'm here to help and support you. Could you please tell me more about what you're going through?"

# Streamlit App
def main():
    st.set_page_config(
        page_title="Psychiatrist Bot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #667eea;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Psychiatrist Bot</h1>', unsafe_allow_html=True)
    st.markdown("### Welcome to your personal mental health companion")
    st.markdown("I'm here to listen, support, and help you through your journey.")
    
    # Load model and data
    model, words, classes, intents = load_model_data()
    if not intents:
        intents = load_intents_only()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Psychiatrist Bot, your personal mental health AI assistant. How are you feeling today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><strong>Psychiatrist Bot:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get bot response
        response = chatbot_response(prompt, model, words, classes, intents)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to display new messages
        st.rerun()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### About Psychiatrist Bot")
        st.markdown("""
        This AI assistant provides mental health support and information. 
        
        **Features:**
        - Anxiety support
        - Depression help
        - Stress management
        - Crisis support
        - General mental health guidance
        
        **Important:** This bot is not a substitute for professional mental health care. 
        If you're in crisis, please contact emergency services or call 988.
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm Psychiatrist Bot, your personal mental health AI assistant. How are you feeling today?"}
            ]
            st.rerun()

if __name__ == "__main__":
    main()