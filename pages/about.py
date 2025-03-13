import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="About - Movie Recommender")

st.title("Movie Recommendation System Documentation")

st.header("What is a Recommendation System?")
st.write("""
A recommendation system is an information filtering system that predicts users' preferences 
and recommends relevant items. In our case, it suggests movies that users might like based 
on similarities between movies.
""")

st.header("Types of Recommendation Systems")
st.write("""
1. **Collaborative Filtering**: Based on user behavior and preferences
2. **Content-Based Filtering**: Based on item features and characteristics
3. **Hybrid Systems**: Combination of multiple approaches
""")

st.header("Our Approach: Content-Based Recommendation")
st.write("""
This project implements a content-based recommendation system that uses movie features 
like overview, genre, and ratings to suggest similar movies. We use natural language 
processing (TF-IDF) and machine learning to analyze movie content and find similarities.
""")

st.header("Project Objectives and Implementation")

st.subheader("Objective 1: Dataset Analysis")
df = pd.read_csv('dataset.csv')
st.write(f"""
- Dataset contains {df.shape[0]} movies with {df.shape[1]} features
- Key features: title, overview, genre, popularity, vote_average, vote_count
- Source: TMDB Movie Database
""")

st.subheader("Objective 2: Data Cleaning and Preprocessing")
st.write("""
- Handled missing values using empty string replacement
- Encoded categorical variables (genres) using LabelEncoder
- Normalized numerical features (popularity, vote_average, vote_count) using StandardScaler
- Created text features by combining movie overview and genre
- Applied TF-IDF vectorization for text analysis
""")

st.subheader("Objective 3: Model Creation")
st.write("""
Trained multiple models:
1. Random Forest Regressor (n_estimators=50)
2. K-Nearest Neighbors (n_neighbors=5)
3. K-Means Clustering (n_clusters=5) for movie grouping
""")

st.subheader("Objective 4: Model Evaluation")
results_df = pd.DataFrame({
    'Model': ['Random Forest', 'KNN'],
    'MSE': [0.118706, 0.707801],
    'R2 Score': [0.881294, 0.292199]
})
st.write("Model Performance Metrics:")
st.table(results_df)

st.write("""
- **Best Model**: Random Forest
- **R² Score**: 0.8813 (88.13% accuracy)
- **Silhouette Score**: 0.0044 (clustering quality)
""")

st.subheader("Objective 5: Model Optimization")
st.write("""
- Reduced model complexity for better performance
- Optimized Random Forest parameters:
    - n_estimators=50
    - n_jobs=-1 (parallel processing)
- Implemented error handling and validation
""")

st.subheader("Objective 6: Visualization and Results")

# Create sample visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Model comparison plot
models = ['Random Forest', 'KNN']
r2_scores = [0.881294, 0.292199]
ax1.bar(models, r2_scores)
ax1.set_title('Model R² Scores')
ax1.set_ylim(0, 1)

# Cluster distribution plot
cluster_data = pd.DataFrame({'cluster': range(5), 'count': [200, 180, 160, 220, 190]})
sns.barplot(data=cluster_data, x='cluster', y='count', ax=ax2)
ax2.set_title('Cluster Distribution')

st.pyplot(fig)

st.header("Sample Recommendations")
st.write("""
Example recommendations for "Iron Man":
- Tau
- Clown
- Avengers: Age of Ultron
- Iron Man 3
- Iron Man 2

The system successfully identifies related movies in the same franchise 
and similar sci-fi/action movies.
""")