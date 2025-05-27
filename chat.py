import os
import nltk
import ssl
import streamlit as st 
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
    
with open("intents.json", "r") as file:
    data = json.load(file)

intents = data["intents"]

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


#this section preprares the intents and train a ML model for the chatbot
#create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

#preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
        
#training model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x,y)

#a function to chat with the chatbot
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        return "I'm sorry, I don't undertand."

counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter")
    
    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")
    
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response{counter}")
        
        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()
            
print(len(patterns), len(tags))

if __name__ == '__main__':
    main()