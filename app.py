import streamlit as st
st.set_page_config(layout="wide", page_title="Movie Recommender", page_icon="🎬")

import pickle
import requests
import numpy as np
from functools import lru_cache
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Move all function definitions to the top
@lru_cache(maxsize=1000)
def fetch_poster(movie_id):
     url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(movie_id)
     data=requests.get(url)
     data=data.json()
     poster_path = data['poster_path']
     full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
     return full_path

@st.cache_data
def load_data():
    movies = pickle.load(open("movies_list.pkl", 'rb'))
    vector = pickle.load(open("vector.pkl", 'rb'))
    best_model = pickle.load(open("best_model.pkl", 'rb'))
    return movies, vector, best_model

@st.cache_data
def get_recommendations(_movies, _vector, movie_title):
    movie_vector = _vector[_movies[_movies['title'] == movie_title].index[0]]
    similarities = cosine_similarity(movie_vector, _vector)
    distances = similarities[0]
    movie_indices = np.argsort(distances)[-6:-1][::-1]
    
    recommend_movie = []
    recommend_poster = []
    for idx in movie_indices:
        movies_id = _movies.iloc[idx].id
        recommend_movie.append(_movies.iloc[idx].title)
        recommend_poster.append(fetch_poster(movies_id))
    return recommend_movie, recommend_poster

# Load data
movies, vector, best_model = load_data()
movies_list = movies['title'].values

# Add custom CSS
st.markdown("""
<style>
.movie-container {
    padding: 1rem;
    border-radius: 15px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    text-align: center;
    height: 100%;
    transition: transform 0.3s;
}
.movie-container:hover {
    transform: translateY(-5px);
}
.movie-title {
    margin-top: 0.8rem;
    font-size: 1.1rem;
    font-weight: bold;
    color: #1E88E5;
}
</style>
""", unsafe_allow_html=True)

# UI Elements
st.title("🎬 Movie Recommender System")
st.write("Find your next favorite movie!")

# Create the selection and button
col1, col2 = st.columns([3, 1])
with col1:
    selectvalue = st.selectbox("🔍 Search for a movie", movies_list)
with col2:
    st.write('')
    st.write('')
    show_button = st.button("🎯 Get Recommendations", use_container_width=True)

# Now we can use show_button
if show_button:
    with st.spinner('Finding recommendations...'):
        movie_name, movie_poster = get_recommendations(movies, vector, selectvalue)
    
    st.subheader("Recommended Movies")
    cols = st.columns(5)
    for idx, (name, poster) in enumerate(zip(movie_name, movie_poster)):
        with cols[idx]:
            st.markdown(f"""
            <div class="movie-container">
                <img src="{poster}" style="width:100%;border-radius:10px;">
                <div class="movie-title">{name}</div>
            </div>
            """, unsafe_allow_html=True)


import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_model_and_data():
    try:
        movies = pickle.load(open('movies_list.pkl', 'rb'))
        vector = pickle.load(open('vector.pkl', 'rb'))
        best_model = pickle.load(open('best_model.pkl', 'rb'))
        return movies, vector, best_model
    except Exception as e:
        print(f"Error loading model data: {str(e)}")
        return None, None, None

def recommend_movies(movie_title, movies, vector, model):
    try:
        if movie_title not in movies['title'].values:
            return []
        
        # Get movie index
        index = movies[movies['title'] == movie_title].index[0]
        
        # Calculate similarity
        similarity = cosine_similarity(vector)
        
        # Get similar movies
        distances = similarity[index]
        movie_indices = np.argsort(distances)[-6:-1][::-1]  # Get top 5 similar movies
        
        recommendations = []
        for idx in movie_indices:
            movie_data = movies.iloc[idx]
            # Use the best model to predict rating
            features = vector[idx].toarray()
            predicted_rating = model.predict(features)[0]
            
            recommendations.append({
                'title': movie_data['title'],
                'predicted_rating': round(predicted_rating, 2)
            })
        
        return recommendations
    
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return []

def main():
    print("Loading model and data...")
    movies, vector, best_model = load_model_and_data()
    
    if movies is None:
        print("Failed to load necessary data!")
        return
    
    print("\n=== Movie Recommendation System ===")
    print("(Using Random Forest Model)")
    
    while True:
        movie_title = input("\nEnter a movie name (or 'quit' to exit): ")
        
        if movie_title.lower() == 'quit':
            break
        
        recommendations = recommend_movies(movie_title, movies, vector, best_model)
        
        if recommendations:
            print("\nRecommended Movies:")
            for rec in recommendations:
                print(f"- {rec['title']} (Predicted Rating: {rec['predicted_rating']})")
        else:
            print("Movie not found or error in generating recommendations.")

if __name__ == "__main__":
    main()
