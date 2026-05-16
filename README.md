---
title: Movie Recommendation System
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.41.1
app_file: app_v3.py
pinned: false
---

# Movie Recommendation System

This project is a movie recommendation system built with the MovieLens 32M dataset.

It includes:

- Content-based filtering
- Item-based collaborative filtering
- User-based collaborative filtering
- Weighted collaborative filtering
- Normalized weighted collaborative filtering
- Hybrid recommendation system
- Streamlit web interface

## Dataset

The dataset files are not included in this repository because they are large.

The application automatically downloads the MovieLens 32M dataset from GroupLens on first launch if these files are missing:

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

The notebooks are useful for development and explanation, but they are not required to run the Streamlit app.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app_v3.py
```

## Deploy on Hugging Face Spaces

Create a new Hugging Face Space with:

```text
SDK: Streamlit
App file: app_v3.py
```

Then upload/push these files:

```text
app_v3.py
requirements.txt
README.md
notebooks/
```

The dataset will be downloaded automatically when the app starts.

## Project Structure

```text
movie-recommendation-system/
│
├── app_v3.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebooks/
│   ├── 1_data_processing.ipynb
│   ├── 2_content_based_filtering.ipynb
│   ├── 3_collaborative_filtering.ipynb
│   └── 4_hybrid_filtering.ipynb
│
└── data/
    └── processed_movies.csv  # auto-generated
```

## Notes

- The first launch can take several minutes because the app downloads and prepares the MovieLens 32M dataset.
- Large dataset files are ignored by Git and should not be uploaded to GitHub.
- The full 32M dataset may require significant memory during model preparation.
