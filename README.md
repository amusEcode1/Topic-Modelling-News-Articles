# 🔍 News Topic Modelling App

This project is a **Natural Language Processing (NLP)** application that performs **topic modelling** on news articles using **LDA (Latent Dirichlet Allocation)** and **NMF (Non-negative Matrix Factorization)**. It helps automatically discover hidden themes and categorize news content.

---

## 🧩 Key Steps

- **Data Handling & Preprocessing**  
  - Tokenization, lowercasing, stopword removal, lemmatization  
  - Cleaning text using `BeautifulSoup` and `re`  

- **Feature Extraction**  
  - Using **CountVectorizer** or **TF-IDF Vectorizer**  

- **Model Training & Topic Modelling**  
  - Latent Dirichlet Allocation (LDA)  
  - Non-negative Matrix Factorization (NMF)  
  - Compare performance between LDA and NMF  

- **Visualization**  
  - Top words per topic (WordClouds)  
  - Interactive topic-word visualization using **pyLDAvis**  

- **Deployment**  
  - Built a **Streamlit app** for interactive prediction  
  - Users can select LDA or NMF and input news articles to get predicted topics

---

## 📂 Dataset

The recommended dataset is the **BBC News Dataset** (available on Kaggle).  

- Discover hidden topics or themes in news articles or blog posts  
- Preprocess text: tokenization, lowercasing, stopword removal  
- Apply LDA and NMF to extract dominant topics  
- Visualize topic-word distributions with **pyLDAvis** or **WordClouds**

- Available on:  
  - [Kaggle - BBC News Dataset](https://www.kaggle.com/datasets/)

---

## 📊 Visualizations

- WordClouds for the most significant words per topic  
- Interactive pyLDAvis visualizations for topic-word distributions  
- Compare LDA vs NMF for dominant topic extraction  

### Example WordCloud
![WordCloud Example](images/wordcloud_example.png)  

### Example pyLDAvis Visualization
![pyLDAvis Example](images/pyldavis_example.png)  

---

## 🧠 Tech Stack & Tools

- **Python Libraries**:  
  `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `WordCloud`, `BeautifulSoup`, `NLTK`, `Scikit-learn`, `pyLDAvis`  
- **Deployment**: Streamlit for interactive prediction  
- **Others**: GitHub / Google Colab / Kaggle for experimentation

---

## 📦 Dependencies

Before running locally, ensure these are installed:

```sh
pip install streamlit numpy pandas matplotlib seaborn wordcloud beautifulsoup4 nltk scikit-learn pyLDAvis joblib
