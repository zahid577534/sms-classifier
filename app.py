import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not already present
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    clean_tokens = [
        ps.stem(word) for word in tokens
        if word not in stopwords.words('english')
        and word not in string.punctuation
    ]
    return " ".join(clean_tokens)

# Load trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app UI
st.title('📩 Email/SMS Spam Classifier')

# Input from user
input_sms = st.text_input("Enter the message:")

# Predict on button click
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # 1. Preprocess
        transform_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transform_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT Spam.")
