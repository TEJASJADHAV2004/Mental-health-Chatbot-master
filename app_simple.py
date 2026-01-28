import nltk
nltk.download('popular')
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # Clean and process the sentence
    sentence_words = clean_up_sentence(sentence)
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.15  # Further lowered threshold
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    # Debug information
    print(f"Input: {sentence}")
    print(f"Top predictions: {return_list[:3] if return_list else 'None above threshold'}")
    
    return return_list

def getResponse(ints, intents_json):
    if ints: 
        tag = ints[0]['intent']
        probability = float(ints[0]['probability'])
        print(f"Matched intent: {tag} with probability: {probability}")
        
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                return result
        return "I understand you're reaching out. Can you tell me more about what's on your mind?"
    else:
        # More contextually appropriate fallback responses
        fallback_responses = [
            "I'm here to listen and support you. Can you tell me more about what's on your mind?",
            "I want to help you. Could you share more about how you're feeling or what you're going through?",
            "I'm here for you. What would you like to talk about today?",
            "Thank you for reaching out. Can you help me understand what you're experiencing?"
        ]
        print("No intent matched - using fallback response")
        return random.choice(fallback_responses)

def chatbot_response(msg):
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
            
        # Use the original ML model for other cases
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        return res
        
    except Exception as e:
        print(f"Error in chatbot_response: {e}")
        return "I'm here to help and support you. Could you please tell me more about what you're going through?"

from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("User message: " + userText)
    
    chatbot_response_text = chatbot_response(userText)
    print("Bot response: " + chatbot_response_text)
    
    return chatbot_response_text

if __name__ == "__main__":
    print("Starting Mental Health Chatbot...")
    print("The chatbot is ready to help with mental health support.")
    app.run(debug=True, host='0.0.0.0', port=5000)