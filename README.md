# Movie Recommendation System

A movie recommendation system built with the MovieLens 32M dataset.

The project includes:

- Content-based filtering
- Item-based collaborative filtering
- User-based collaborative filtering
- Weighted collaborative filtering
- Normalized weighted collaborative filtering
- Hybrid recommendation system
- Streamlit web interface

## Live Demo

Direct app link:

https://ttrau-movie-recommendation-system.hf.space

## Dataset

This project uses the MovieLens 32M dataset.

The dataset files are **not included** in this repository because they are large.

The application automatically downloads the required MovieLens 32M files on first launch if they are missing:

```text
data/movies.csv
data/ratings.csv
data/tags.csv
```

The application also automatically creates:

```text
data/processed_movies.csv
```

from `movies.csv` and `tags.csv` if it does not already exist.

Therefore, users do **not** need to run the notebooks manually to use the application.

## Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app_v3.py
```

The first launch can take several minutes because the application downloads and prepares the MovieLens 32M dataset.

## Project Structure

```text
movie-recommendation-system/
│
├── app_v3.py
├── requirements.txt
├── README.md
├── .gitignore
├── Dockerfile
│
├── notebooks/
│   ├── 1_data_processing.ipynb
│   ├── 2_content_based_filtering.ipynb
│   ├── 3_collaborative_filtering.ipynb
│   └── 4_hybrid_filtering.ipynb
│
└── data/
    └── .gitkeep
```

## Recommendation Methods

### Content-Based Filtering

Uses movie metadata such as title, genres, and tags to recommend similar movies.

### Item-Based Collaborative Filtering

Uses user rating behavior to recommend movies that were rated similarly by users.

### User-Based Collaborative Filtering

Finds users with similar rating patterns and recommends movies based on their preferences.

### Weighted Collaborative Filtering

Uses similarity scores as weights when calculating recommendation scores from similar users.

### Normalized Weighted Collaborative Filtering

Normalizes user ratings by subtracting each user's mean rating before calculating weighted predictions.

### Hybrid Filtering

Combines content-based and item-based collaborative filtering scores into a final hybrid recommendation score.

The hybrid system dynamically adjusts the contribution of content-based and collaborative filtering according to the selected movie's rating count.

## Notes

- Large dataset files are ignored by Git and should not be uploaded to GitHub.
- The first deployment startup may take several minutes.
- If the deployment environment restarts, the dataset may need to be downloaded again.
- The notebooks are included for development and explanation purposes.
