import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
lemmatizer = nltk.stem.WordNetLemmatizer()
import random

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

@st.cache_data()
def load_data(text):
    data = pd.read_csv(text, sep= ',')
    return data

df = pd.read_csv('cleaned_moneypay.txt')

def preprocess_text(text):
    sentences =nltk.sent_tokenize(text)

    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


df['tokenized Customer'] = df['Customer'].apply(preprocess_text)
xtrain = df['tokenized Customer'].to_list()


tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)

#*************************************Streamlit operations************************************************

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-size: 60px; font-family: Helvetica'>WELCOME TO MONEYPAY</h1>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Kunle Odukoya</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)
st.image('pngwing.com (14).png', width = 400,  caption = "YOUR PARTNER IN PROGRESS")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

user_history = []
bot_history = []

missy_image, space1,space2, chats = st.columns(4)
with missy_image:
    missy_image.image('pngwing.com (15).png', width = 300)

with chats:
    user_message = chats.text_input('Hello, you can ask your question here: ')
    def responder(test):
        user_input_processed = preprocess_text(test)
        vectorized_user_input = tfidf_vectorizer.transform([user_input_processed])
        similarity_score = cosine_similarity(vectorized_user_input, corpus)
        argument_maximum = similarity_score.argmax()

        chats.write(df['Response'].iloc[argument_maximum])

bot_greetings = ['Hello user, Welcome to Moneypay, I am Missy, How can I help you?',
    'Welcome to MoneyPay, I am Missy, How are you today? How may I be of service to you?',
    'Missy here, This is MoneyPay, is there anything I can do for you?',
    'I am Missy and this is MoneyPay platform, is there anyway I can be of service to you?',
    'Welcome to our MoneyPay platform, My name is Missy, How can I help you?'
             ]

bot_farewell = ['Thanks for checking out MoneyPay, bye',
             'Hope, I have been helpful, bye',
             'Hope to see you soon',
             'Do not hesiatate to contact me again',
             'See ya..bye'
           ]

human_greetings = ['hi', 'hello there', 'hey','hello']
human_exits = ['thanks bye', 'bye', 'quit','bye bye', 'close', 'exit']


random_greeting = random.choice(bot_greetings)
random_farewell = random.choice(bot_farewell)

if user_message.lower() in human_greetings:
    chats.write(f"\nChatbot: {random_greeting}")
    user_history.append(user_message)
    bot_history.append(random_greeting)
elif user_message.lower() in human_exits:
    chats.write(f"\nChatbot: {random_farewell}")
    user_history.append(user_message)
    bot_history.append(random_farewell)
elif  user_message == '':
    chats.write('')
else:
    response = responder(user_message)
    user_history.append(user_message)
    bot_history.append(response)

import csv

with open('user_history.txt', 'a') as file:
    for item in user_history:
        file.write(str(item) + '\n')
    
with open('bot_history.txt', 'a') as file:
    for item in bot_history:
        file.write(str(item) + '\n')


with open('user_history.txt') as f:
    reader = csv.reader(f)
    data1 = list(reader)

with open('bot_history.txt') as f:
    reader = csv.reader(f)
    data2 = list(reader)

data1 = pd.Series(data1)
data2 = pd.Series(data2)

history = pd.DataFrame({'User Input': data1, 'Bot_Reply': data2})

st.subheader('Chat History', divider = True)
st.dataframe(history, use_container_width = True)

