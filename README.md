# Simple Recommendation System Demo

This repository contains a small, self-contained Python demo of two simple recommenders:

- Content-based: TF-IDF on item text (title + description + genres) and item-item cosine similarity.
- User-based collaborative filtering: cosine similarity on user rating vectors and weighted predictions.

Files:
- `recommendation_system.py`: main demo script. Run to see example recommendations.
- `requirements.txt`: Python dependencies.

Quick start (PowerShell):

```powershell
python -m pip install -r requirements.txt
python recommendation_system.py
```

The script generates a tiny synthetic dataset and prints content-based and collaborative recommendations.

Next steps:
- Replace the synthetic dataset with your real items and ratings.
- Add evaluation (train/test split) and more sophisticated models.
# AI_project