import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load dataset
def load_data():
    try:
        df = pd.read_csv('dataset.csv')
        return df
    except FileNotFoundError:
        print("Error: dataset.csv not found in the current directory")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)

def preprocess_data(df):
    try:
        # Handle missing values
        df.fillna('', inplace=True)
        
        # Encode categorical variables
        label_enc = LabelEncoder()
        if 'genre' in df.columns:
            df['genre_encoded'] = label_enc.fit_transform(df['genre'].astype(str))
        
        # Normalize numerical features if available
        num_cols = ['popularity', 'vote_average', 'vote_count']
        for col in num_cols:
            if col in df.columns:
                df[col] = StandardScaler().fit_transform(df[[col]].fillna(0))
        
        return df
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        exit(1)

def create_features(df):
    try:
        df['tags'] = df['overview'].astype(str) + ' ' + df['genre'].astype(str)
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        vector = tfidf.fit_transform(df['tags'].values.astype('U'))
        return df, vector
    except Exception as e:
        print(f"Error in feature creation: {str(e)}")
        exit(1)

# Train multiple models
def train_models(X, y):
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, n_jobs=-1),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    results = {}
    best_model = None
    best_r2 = float('-inf')
    best_model_name = None

    print("Training models:")
    for name, model in models.items():
        print(f"- Training {name}...")
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
        
        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
    
    print(f"\nBest performing model: {best_model_name} (R2 Score: {best_r2:.4f})")
    return results, best_model

# Clustering model
def clustering_model(X):
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, cluster_labels)
    return cluster_labels, silhouette

# Recommendation System
def recommend(movie_title, new_data, similarity):
    if movie_title not in new_data['title'].values:
        return []
    index = new_data[new_data['title'] == movie_title].index[0]
    distances = similarity[index]
    movie_indices = np.argsort(distances)[-6:-1]
    return [new_data.iloc[i].title for i in movie_indices]

if __name__ == "__main__":
    try:
        # Load and preprocess data
        print("Loading data...")
        movies = load_data()
        
        print("Preprocessing data...")
        movies = preprocess_data(movies)
        
        print("Creating features...")
        movies, vector = create_features(movies)

        # Split data for training
        if 'vote_average' in movies.columns:
            print("Training models...")
            y = movies['vote_average']
            X = vector.toarray()
            results, best_model = train_models(X, y)
            cluster_labels, silhouette = clustering_model(X)
            movies['cluster'] = cluster_labels

            # Save processed data
            try:
                pickle.dump(movies, open('movies_list.pkl', 'wb'))
                pickle.dump(vector, open('vector.pkl', 'wb'))
                pickle.dump(results, open('results.pkl', 'wb'))
                pickle.dump(best_model, open('best_model.pkl', 'wb'))
                pickle.dump(silhouette, open('silhouette.pkl', 'wb'))
                print("Model data saved successfully!")
            except Exception as e:
                print(f"Error saving model data: {str(e)}")

        # Print results
        print("\n=== Model Evaluation Results ===")
        df_results = pd.DataFrame(results).T
        print(df_results)
        
        # Display Clustering Results
        print(f"\nSilhouette Score for Clustering: {silhouette:.4f}")
        print("\n=== Cluster Distribution ===")
        plt.figure(figsize=(10, 6))
        sns.countplot(x=movies['cluster'])
        plt.title("Cluster Distribution")
        plt.show()
        
        # Recommendation System
        print("\n=== Movie Recommendations ===")
        while True:
            sample_movie = input("Enter a movie name (or 'quit' to exit): ")
            if sample_movie.lower() == 'quit':
                break
                
            recommendations = recommend(sample_movie, movies, cosine_similarity(vector))
            if recommendations:
                print("\nRecommended Movies:")
                for movie in recommendations:
                    print(f"- {movie}")
            else:
                print("Movie not found.")
            print()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
