"""Simple recommendation system demo.

This file implements two small recommenders:
- Content-based (TF-IDF on item descriptions + cosine similarity)
- User-based collaborative filtering (cosine similarity on ratings)

The module includes a small synthetic dataset and a `run_demo()`
function that prints example recommendations.
"""

from typing import List
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_demo_data():
	items = [
		{"item_id": 1, "title": "The Space Odyssey", "description": "Sci-fi adventure in space", "genres": "sci-fi adventure"},
		{"item_id": 2, "title": "Deep Sea Tales", "description": "Underwater exploration and drama", "genres": "adventure drama"},
		{"item_id": 3, "title": "Romantic Sunset", "description": "A touching romantic story", "genres": "romance"},
		{"item_id": 4, "title": "Galaxy Wars", "description": "Intergalactic battles and heroes", "genres": "sci-fi action"},
		{"item_id": 5, "title": "Cooking with Love", "description": "A chef finds love and flavor", "genres": "romance comedy"},
		{"item_id": 6, "title": "The Last Detective", "description": "Crime noir and a gritty mystery", "genres": "crime mystery"},
	]

	items_df = pd.DataFrame(items)

	# Simple synthetic ratings: user_id, item_id, rating (1-5)
	ratings = [
		(1, 1, 5), (1, 4, 4), (1, 3, 1),
		(2, 2, 4), (2, 1, 5), (2, 4, 5),
		(3, 3, 5), (3, 5, 4), (3, 6, 2),
		(4, 6, 5), (4, 2, 2), (4, 1, 1),
	]
	ratings_df = pd.DataFrame(ratings, columns=["user_id", "item_id", "rating"])

	return items_df, ratings_df


def build_content_recommender(items_df: pd.DataFrame):
	# Combine text features
	corpus = (items_df["title"] + " " + items_df["description"] + " " + items_df["genres"]).tolist()
	tfidf = TfidfVectorizer(stop_words="english")
	X = tfidf.fit_transform(corpus)
	sim = cosine_similarity(X)

	# Map item_id to matrix index
	id_to_idx = {int(i): idx for idx, i in enumerate(items_df["item_id"].tolist())}
	idx_to_id = {v: k for k, v in id_to_idx.items()}

	def recommend(item_ids: List[int], top_n: int = 5) -> List[int]:
		# Compute average similarity of all items to the given liked items
		liked_idx = [id_to_idx[i] for i in item_ids if i in id_to_idx]
		if not liked_idx:
			return []
		scores = sim[liked_idx].mean(axis=0)
		ranked_idx = np.argsort(-scores)
		recommendations = []
		for idx in ranked_idx:
			item_id = idx_to_id[idx]
			if item_id not in item_ids:
				recommendations.append(int(item_id))
			if len(recommendations) >= top_n:
				break
		return recommendations

	return recommend


def build_user_collab_recommender(ratings_df: pd.DataFrame):
	# Create user-item pivot
	pivot = ratings_df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)
	user_ids = pivot.index.tolist()
	item_ids = pivot.columns.tolist()

	user_matrix = pivot.values
	user_sim = cosine_similarity(user_matrix)
	user_id_to_idx = {u: i for i, u in enumerate(user_ids)}

	def recommend(user_id: int, top_n: int = 5) -> List[int]:
		if user_id not in user_id_to_idx:
			return []
		uidx = user_id_to_idx[user_id]
		sims = user_sim[uidx]

		# Weighted sum of other users' ratings
		weights = sims.reshape(1, -1)
		weighted_ratings = weights.dot(user_matrix)
		# Normalize by sum of similarities
		sim_sums = np.abs(sims).sum()
		if sim_sums == 0:
			scores = weighted_ratings.flatten()
		else:
			scores = (weighted_ratings.flatten() / sim_sums)

		# Do not recommend items the user already rated
		user_rated = set(ratings_df[ratings_df["user_id"] == user_id]["item_id"].tolist())
		ranked_indices = np.argsort(-scores)
		recommendations = []
		for idx in ranked_indices:
			item_id = int(item_ids[idx])
			if item_id in user_rated:
				continue
			# Only recommend if score > 0 (some signal)
			if scores[idx] <= 0:
				continue
			recommendations.append(item_id)
			if len(recommendations) >= top_n:
				break
		return recommendations

	return recommend


def run_demo():
	items_df, ratings_df = create_demo_data()

	print("Items:")
	print(items_df[["item_id", "title"]].to_string(index=False))
	print("\nRatings:\n", ratings_df)

	# Content-based: assume user 1 likes item 1
	content_rec = build_content_recommender(items_df)
	liked = [1]
	print(f"\nContent-based recommendations for liked items {liked}:")
	content_ids = content_rec(liked, top_n=3)
	print(items_df[items_df["item_id"].isin(content_ids)][["item_id", "title"]].to_string(index=False))

	# Collaborative: get recommendations for user 3
	collab_rec = build_user_collab_recommender(ratings_df)
	user = 3
	print(f"\nCollaborative recommendations for user {user}:")
	collab_ids = collab_rec(user, top_n=3)
	if collab_ids:
		print(items_df[items_df["item_id"].isin(collab_ids)][["item_id", "title"]].to_string(index=False))
	else:
		print("No collaborative recommendations available.")


if __name__ == "__main__":
	run_demo()





