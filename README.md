<div align="center">

  <a href="https://tarkptel.github.io/">
    <img src="https://img.shields.io/badge/🌐-Portfolio-blue" height="28">
  </a>
  <a href="https://www.kaggle.com/tark01/">
    <img src="https://img.shields.io/badge/-Kaggle-20BEFF?logo=kaggle&logoColor=white" height="28">
  </a>
  <a href="https://www.linkedin.com/in/tark-patel/">
    <img src="https://img.shields.io/badge/-LinkedIn-0077B5?logo=linkedin&logoColor=white" height="28">
  </a>

</div>

<div align="center">

# 🚀 Fake Review Detector 🛍️

</div>

A machine learning-based web application that detects whether a product review is **real** or **fake**, helping users make more informed online purchase decisions. <br><br>



## 🌟 Project Overview

This project is built to detect suspicious or fake product reviews based on their textual content. You simply paste a product review into the app, and the model analyzes it to tell you whether the review is likely to be real or fake. <br><br>


## 🌐 Live Demo

🚧 _Live demo available on [**Hugging Face Spaces**](https://huggingface.co/spaces/tarkpatel/Fake_Review_Detector)_ <br><br>


## 🌟 Model Overview

The model was trained on a labeled dataset of product reviews using Natural Language Processing (NLP) techniques. Features like text length, sentiment polarity, and TF-IDF vectors were used to train a classification model. <br><br>

### 🌟 Features used:
- TF-IDF vectors of the review text
- Text length and word count
- Sentiment polarity and subjectivity
- Custom heuristic features like excessive repetition or extreme sentiment <br><br>


## 🌟 Workflow

*Here’s how the app works behind the scenes:*

1. **Input Review**  
   User pastes a product review into the text box.

2. **Preprocessing**  
   The review is cleaned (punctuation removal, lowercasing, etc.) and processed using:
   - TF-IDF Vectorization  
   - Sentiment Analysis (TextBlob)  
   - Custom Features (length, polarity, etc.)

3. **Prediction**  
   The processed text is passed into the trained ML model, which classifies the review as:
   - ✅ **Real** or  
   - ❌ **Fake**

4. **Result Display**  
   The prediction is displayed instantly on the interface. <br><br>


## 🌟 Tech Stack

- **Python**
- **scikit-learn** for model training
- **NLTK** and **TextBlob** for text processing
- **Streamlit** for building the web app <br><br>


<h2> 👑 Author</h2>
    <p>Developed by <strong>Tark Patel</strong>.</p>
    <p><a href="https://www.linkedin.com/in/tark-patel/" target="_blank">LinkedIn</a> | <a href="https://www.kaggle.com/tark01" target="_blank">Kaggle</a></p>

