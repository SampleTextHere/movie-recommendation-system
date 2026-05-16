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

The dataset files are not included in this repository because they are too large for GitHub.

Download the MovieLens 32M dataset and place these files inside the `data/` folder:

```text
data/movies.csv
data/ratings.csv
data/tags.csv
```

The application automatically creates:

```text
data/processed_movies.csv
```

from `movies.csv` and `tags.csv` if it does not already exist.

So the notebooks are useful for development and explanation, but they are not required to run the Streamlit app.

## Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app_v3.py
```

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
    ├── movies.csv
    ├── ratings.csv
    ├── tags.csv
    └── processed_movies.csv  # auto-generated
```

## Notes

- `processed_movies.csv` is generated automatically by the app.
- Large dataset files are ignored by Git and should not be uploaded to GitHub.
- The notebooks are included for documentation and development purposes.
