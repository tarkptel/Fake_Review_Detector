import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import re
import pickle
from textblob import TextBlob
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')


# Load your models
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

    

# Clean Text
def clean_text(Text):
   
    Text = str(Text).lower() # convert to lowercase
    Text = re.sub('\[.*?\]', '', Text) 
    Text = re.sub('https?://\S+|www\.\S+', '', Text) # Remove URls
    Text = re.sub('<.*?>+', '', Text)
    Text = re.sub(r'[^a-z0-9\s]', '', Text) # Remove punctuation
    Text = re.sub('\n', '', Text)
    Text = re.sub('\w*\d\w*', '', Text)


    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = Text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back to string
    cleaned_text = " ".join(tokens)
    
    return cleaned_text

#create feature

def extract_features(text):
    blob = TextBlob(text)
    words = text.split()
    num_words = len(words)
    
    return {
        "review_length": len(text),
        "avg_word_length": sum(len(w) for w in words) / (num_words + 1),
     #   "num_exclamations": text.count("!"),
      #  "num_uppercase_words": sum(1 for w in words if w.isupper()),
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "repetition_ratio": len(set(words)) / (num_words + 1),
    }


# Streamlit UI
st.title("🕵️ Fake Review Detector")
st.markdown("Paste a product review and we'll detect if it's **Fake** or **Original**.")

user_input = st.text_area("✍️ Enter your review:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:

        # Clean Text
        text = clean_text(user_input)
        
        # Create Features
        features_df = extract_features(user_input)
    
        #Create DataFrame
        features_df = pd.DataFrame([features_df])
        
        #Text to Num
        text_vector = vectorizer.transform([user_input])
    
        #Combine all Inputs
        final_input = csr_matrix(features_df.values)   # Convert to sparse format
        final_input = hstack([final_input,text_vector]) 
    
        # Finel Prediction
        prediction = model.predict(final_input)[0]
        label = "✅ Original Review" if prediction == 1 else "🚨 Fake Review"
        
        st.subheader("Result:")
        st.success(label)
