# 🎬 Movie Recommendation System

A content-based movie recommender built with Streamlit, leveraging machine learning and natural language processing (NLP) to suggest movies similar to user preferences.

## 📌 What is a Recommendation System?

A recommendation system filters and predicts a user’s preferences to suggest relevant items. In this project, the system recommends movies based on their content and similarity to other titles.

### 💡 Types of Recommendation Systems

- Collaborative Filtering
Recommends items based on user behavior and preference patterns.

- Content-Based Filtering
Suggests items similar to what a user likes, based on features like genre or description.

- Hybrid Models
Combines collaborative and content-based approaches for improved accuracy.

### 🚀 Our Approach: Content-Based Filtering

We use natural language processing (NLP) with TF-IDF vectorization on movie overviews and genres to recommend similar titles. The system analyzes movie content to identify thematic or narrative similarities.

## 🎯 Project Objectives

### ✅ Objective 1: Dataset Analysis

- Loaded dataset with 10,000 movies and 9 features

- Key features: title, overview, genre, popularity, vote_average, vote_count

Source: TMDB API dataset

### ✅ Objective 2: Data Cleaning & Preprocessing

- Handled missing values (replaced with empty strings)

- Encoded categorical variables (genres)

- Normalized numerical columns

- Combined overview and genre to form text features

- Applied TF-IDF Vectorization for text feature extraction

### ✅ Objective 3: Model Building

- Implemented and evaluated multiple models:

- Random Forest Regressor (n_estimators=50)

- K-Nearest Neighbors (n_neighbors=5)

- K-Means Clustering (n_clusters=5) for unsupervised grouping

### ✅ Objective 4: Model Evaluation

- Random Forest - MSE Score: 0.118706, R² Score: 0.8813

- KNN - MSE Score: 0.707801, R² Score: 0.2922

- 📈 Best Model: Random Forest

- 🧠 Accuracy (R² Score): 88.13%

### ✅ Objective 5: Model Optimization

- Tuned hyperparameters for efficiency:

- n_estimators=50, n_jobs=-1

- Simplified model structure for deployment

- Added error handling and data validation

### 🎥 Sample Recommendations

- For the movie “Iron Man”, the system recommends:

Tau

Clown

Avengers: Age of Ultron

Iron Man 3

Iron Man 2

- These recommendations reflect genre similarity and franchise alignment, demonstrating successful content-based matching.

### 📦 Technologies Used

- Python

- Pandas, NumPy

- Scikit-learn

- Matplotlib, Seaborn

- Streamlit

- NLP (TF-IDF)

### 🛠️ How to Run

```python
pip install streamlit pandas matplotlib seaborn scikit-learn
streamlit run app.py
```

- Make sure dataset.csv is in the same directory.

🙌 Contributions
Pull requests and suggestions are welcome!
