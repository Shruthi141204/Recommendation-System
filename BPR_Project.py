import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import os
import time

class BPR:
    def __init__(self, num_users, num_items, learning_rate=0.01, reg_param=0.01, num_iterations=5):
        self.num_users = num_users
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.num_iterations = num_iterations
        self.user_embeddings = np.random.normal(size=(num_users, 10))
        self.item_embeddings = np.random.normal(size=(num_items, 10))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, train_data):
        for _ in range(self.num_iterations):
            for row in train_data.itertuples(index=False):
                user, item_i, item_j = row[0], row[1], row[1]

                # Calculate the difference in scores
                diff = np.dot(self.user_embeddings[user], self.item_embeddings[item_i]) - np.dot(self.user_embeddings[user], self.item_embeddings[item_j])

                # Update user and item embeddings
                self.user_embeddings[user] += self.learning_rate * (1 - self._sigmoid(diff)) * (self.item_embeddings[item_j] - self.item_embeddings[item_i]) - self.reg_param * self.user_embeddings[user]
                self.item_embeddings[item_i] += self.learning_rate * (1 - self._sigmoid(diff)) * self.user_embeddings[user] - self.reg_param * self.item_embeddings[item_i]
                self.item_embeddings[item_j] += self.learning_rate * (1 - self._sigmoid(diff)) * (-self.user_embeddings[user]) - self.reg_param * self.item_embeddings[item_j]

    def predict(self, user_id, item_id):
        return np.dot(self.user_embeddings[user_id], self.item_embeddings[item_id])

# Load ratings and movies data
# Adjust the file paths if needed
ratings_file = 'D:/AMRITA/Sem4/ML/Final/ratings.csv'
movies_file = 'D:/AMRITA/Sem4/ML/Final/movies.csv'

if not os.path.exists(ratings_file):
    raise FileNotFoundError(f"File not found: {ratings_file}")

if not os.path.exists(movies_file):
    raise FileNotFoundError(f"File not found: {movies_file}")

ratings_data = pd.read_csv(ratings_file)
movies_data = pd.read_csv(movies_file)

# Map movie IDs to sequential integers
movie_id_map = {movie_id: i for i, movie_id in enumerate(sorted(ratings_data['movieId'].unique()))}

# Convert user and movie IDs to sequential integers
ratings_data['userId'] = ratings_data['userId'].astype('category').cat.codes
ratings_data['movieId'] = ratings_data['movieId'].map(movie_id_map)

# Number of users and items
num_users = len(ratings_data['userId'].unique())
num_items = len(movie_id_map)

# Shuffle ratings data
ratings_data = ratings_data.sample(frac=1).reset_index(drop=True)

# Split data into training and test sets (80-20 split)
train_size = int(0.8 * len(ratings_data))
train_data = ratings_data[:train_size]
test_data = ratings_data[train_size:]

# Initialize and train BPR model
bpr_model = BPR(num_users, num_items)
start_train_time = time.time()
bpr_model.fit(train_data)
end_train_time = time.time()
bpr_run_time = end_train_time - start_train_time

def get_recommendations(user_id, top_n=5):
    # Calculate predicted scores for all items
    predicted_scores = []
    for item_id in range(num_items):
        score = bpr_model.predict(user_id, item_id)
        predicted_scores.append(score)

    # Get the indices of the top N items with the highest scores
    top_n_indices = np.argsort(predicted_scores)[-top_n:][::-1]

    # Map back to original movie IDs and get movie details
    recommended_movie_ids = [list(movie_id_map.keys())[list(movie_id_map.values()).index(i)] for i in top_n_indices]
    recommended_movies = movies_data[movies_data['movieId'].isin(recommended_movie_ids)]

    return recommended_movies

def recommend_movies():
    user_id = int(user_id_entry.get())
    start_time = time.time()
    recommended_movies = get_recommendations(user_id)
    end_time = time.time()
    run_time = end_time - start_time
    result_text = f"Top 5 Recommended Movies (Runtime: {run_time:.2f} seconds):\n"
    for index, row in recommended_movies.iterrows():
        result_text += f"{row['title']}\n"
    result_text += f"\nBPR Model Training Time: {bpr_run_time:.2f} seconds"
    result_label.config(text=result_text, font=("Helvetica", 10, "bold"))

# Create the main window
root = tk.Tk()
root.title("Movie Recommendation System")

# Create and place the components
ttk.Label(root, text="Enter User ID:").grid(row=0, column=0, padx=10, pady=10)
user_id_entry = ttk.Entry(root)
user_id_entry.grid(row=0, column=1, padx=10, pady=10)

recommend_button = ttk.Button(root, text="Recommend Movies", command=recommend_movies)
recommend_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

result_label = ttk.Label(root, text="")
result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Start the main event loop
root.mainloop()
