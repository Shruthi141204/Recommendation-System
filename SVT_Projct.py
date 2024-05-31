import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import time
from sklearn.neighbors import NearestNeighbors

# Load movies and ratings data
movies_file = 'D:/AMRITA/Sem4/ML/Final/movies.csv'
ratings_file = 'D:/AMRITA/Sem4/ML/Final/ratings.csv'

movies_df = pd.read_csv(movies_file)
ratings_df = pd.read_csv(ratings_file)

# Prepare the data
data = {
    'user_id': list(ratings_df['userId']),
    'movie_id': list(ratings_df['movieId']),
    'rating': list(ratings_df['rating'])
}

df = pd.DataFrame(data)

# Convert the dataframe to a user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Mask for observed entries
mask = user_item_matrix > 0

# Matrix completion with SVT
def svt_matrix_completion(X, mask, tau=1.0, delta=1.0, max_iter=5, tol=1e-4):
    m, n = X.shape
    Y = np.zeros((m, n))
    for i in range(max_iter):
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        S_threshold = np.maximum(S - tau, 0)
        X_hat = np.dot(U, np.dot(np.diag(S_threshold), Vt))
        Y = Y + delta * (mask * (X - X_hat))
        if np.linalg.norm(mask * (X - X_hat), 'fro') < tol:
            break
    return X_hat

X = user_item_matrix.values

# Measure runtime of SVT matrix completion
start_time_svt = time.time()
X_completed = svt_matrix_completion(X, mask.values)
end_time_svt = time.time()
svt_runtime = end_time_svt - start_time_svt

completed_matrix = pd.DataFrame(X_completed, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Fit kNN model on completed_matrix
k = 5  # Number of neighbors
knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
knn_model.fit(completed_matrix)

# Function to get k nearest neighbors for a given user
def get_k_nearest_neighbors(user_id):
    user_ratings = completed_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_ratings)
    return distances, indices

# Get recommendations from k nearest neighbors
def get_recommendations_from_neighbors(indices, num_recommendations):
    recommendations = []
    for neighbor_index in indices.flatten():
        neighbor_ratings = completed_matrix.iloc[neighbor_index]
        top_recommendations = neighbor_ratings.sort_values(ascending=False).index[:num_recommendations]
        recommendations.extend(top_recommendations)
    return list(pd.Series(recommendations).unique())[:num_recommendations]


def recommend_movies():
    user_id = int(user_id_entry.get())
    
    # Measure runtime of kNN recommendation
    start_time_knn = time.time()
    distances, indices = get_k_nearest_neighbors(user_id)
    recommended_movie_ids = get_recommendations_from_neighbors(indices, 5)
    end_time_knn = time.time()
    knn_runtime = end_time_knn - start_time_knn
    
    # Map movie IDs to movie titles
    recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
    recommended_movie_titles = recommended_movies['title'].tolist()
    
    # Display results
    result_text = f"Top 5 Recommended Movies:\n"
    for title in recommended_movie_titles:
        result_text += f"{title}\n"
    result_text += f"\nSVT Runtime: {svt_runtime:.2f} seconds\n"
    result_text += f"kNN Recommendation Runtime: {knn_runtime:.2f} seconds"
    
    result_label.config(text=result_text)

# Create the main window
root = tk.Tk()
root.title("Movie Recommendation System")

# Create and place the components
ttk.Label(root, text="Enter User ID:", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=10, pady=10)
user_id_entry = ttk.Entry(root, font=("Helvetica", 10))
user_id_entry.grid(row=0, column=1, padx=10, pady=10)

recommend_button = ttk.Button(root, text="Recommend Movies", command=recommend_movies, style="Big.TButton")
recommend_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

result_label = ttk.Label(root, text="", font=("Helvetica", 10))
result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Customize button style
style = ttk.Style()
style.configure("Big.TButton", font=("Helvetica", 10, "bold"))

# Start the main event loop
root.mainloop()
