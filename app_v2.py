from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Page setup
# -------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

st.title("Movie Recommendation System")
st.caption("APP VERSION: user-analysis-v2")

st.write(
    "This application recommends movies using content-based filtering, "
    "item-based collaborative filtering, hybrid filtering, and user-based "
    "collaborative filtering."
)


# -------------------------------
# File paths
# -------------------------------

def get_data_path(filename):
    data_path = Path("data") / filename

    if data_path.exists():
        return data_path

    current_path = Path(filename)

    if current_path.exists():
        return current_path

    raise FileNotFoundError(
        f"Could not find {filename}. Expected data/{filename} or {filename}."
    )


# -------------------------------
# Load datasets
# -------------------------------

@st.cache_data
def load_data():
    movies = pd.read_csv(
        get_data_path("processed_movies.csv")
    )

    ratings = pd.read_csv(
        get_data_path("ratings.csv")
    )

    movies["metadata"] = (
        movies["metadata"]
        .fillna("")
        .astype(str)
    )

    return movies, ratings


with st.spinner("Loading datasets..."):
    movies, ratings = load_data()


# -------------------------------
# Prepare models
# -------------------------------

@st.cache_resource
def prepare_models(movies, ratings):

    # Content-based filtering
    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(
        movies["metadata"]
    )

    content_similarity = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    content_indices = pd.Series(
        movies.index,
        index=movies["title"]
    ).drop_duplicates()


    # Item-based collaborative filtering
    movie_user_matrix = ratings.pivot_table(
        index="movieId",
        columns="userId",
        values="rating"
    )

    movie_user_filled = movie_user_matrix.fillna(0)

    movie_similarity = cosine_similarity(
        movie_user_filled
    )

    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index
    )

    movie_title_to_id = pd.Series(
        movies["movieId"].values,
        index=movies["title"]
    ).drop_duplicates()

    movie_rating_counts = ratings.groupby("movieId").size()


    # User-based collaborative filtering
    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    user_movie_filled = user_movie_matrix.fillna(0)

    user_similarity = cosine_similarity(
        user_movie_filled
    )

    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )


    # Normalized user-based collaborative filtering
    user_mean_ratings = user_movie_matrix.mean(axis=1)

    normalized_matrix = user_movie_matrix.sub(
        user_mean_ratings,
        axis=0
    )

    normalized_filled = normalized_matrix.fillna(0)

    normalized_similarity = cosine_similarity(
        normalized_filled
    )

    normalized_similarity_df = pd.DataFrame(
        normalized_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )

    return {
        "tfidf": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "content_similarity": content_similarity,
        "content_indices": content_indices,
        "movie_user_matrix": movie_user_matrix,
        "movie_user_filled": movie_user_filled,
        "movie_similarity": movie_similarity,
        "movie_similarity_df": movie_similarity_df,
        "movie_title_to_id": movie_title_to_id,
        "movie_rating_counts": movie_rating_counts,
        "user_movie_matrix": user_movie_matrix,
        "user_movie_filled": user_movie_filled,
        "user_similarity": user_similarity,
        "user_similarity_df": user_similarity_df,
        "user_mean_ratings": user_mean_ratings,
        "normalized_matrix": normalized_matrix,
        "normalized_filled": normalized_filled,
        "normalized_similarity": normalized_similarity,
        "normalized_similarity_df": normalized_similarity_df,
    }


with st.spinner("Preparing recommendation models..."):
    models = prepare_models(movies, ratings)


# expose variables with notebook-like names
tfidf = models["tfidf"]
tfidf_matrix = models["tfidf_matrix"]
content_similarity = models["content_similarity"]
content_indices = models["content_indices"]

movie_user_matrix = models["movie_user_matrix"]
movie_user_filled = models["movie_user_filled"]
movie_similarity = models["movie_similarity"]
movie_similarity_df = models["movie_similarity_df"]
movie_title_to_id = models["movie_title_to_id"]
movie_rating_counts = models["movie_rating_counts"]

user_movie_matrix = models["user_movie_matrix"]
user_movie_filled = models["user_movie_filled"]
user_similarity = models["user_similarity"]
user_similarity_df = models["user_similarity_df"]

user_mean_ratings = models["user_mean_ratings"]
normalized_matrix = models["normalized_matrix"]
normalized_filled = models["normalized_filled"]
normalized_similarity = models["normalized_similarity"]
normalized_similarity_df = models["normalized_similarity_df"]


# -------------------------------
# Helper functions
# -------------------------------

def find_movie(title):
    matches = movies[
        movies["title"]
        .str.lower()
        .str.contains(title.lower(), regex=False)
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def shorten_metadata(df):
    output = df.copy()

    if "metadata" in output.columns:
        output["metadata"] = (
            output["metadata"]
            .astype(str)
            .str.slice(0, 140)
        )

    return output


# -------------------------------
# Movie-based recommenders
# -------------------------------

def content_recommend(
    title,
    n_movies=10,
    round=4
):
    movie = find_movie(title)

    if movie is None:
        return "Movie not found."

    title = movie["title"]

    idx = content_indices[title]

    sim_scores = list(
        enumerate(content_similarity[idx])
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:n_movies+1]

    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    result = movies[
        ["movieId", "title", "metadata"]
    ].iloc[movie_indices].copy()

    result["content_score"] = scores

    result["content_score"] = (
        result["content_score"]
        .round(round)
    )

    return result.reset_index(drop=True)


def collaborative_recommend(
    title,
    n_movies=10,
    min_ratings=20,
    round=4
):
    movie = find_movie(title)

    if movie is None:
        return "Movie not found."

    title = movie["title"]
    movie_id = movie["movieId"]

    if movie_id not in movie_similarity_df.columns:
        return "This movie has no rating data."

    similar_movies = (
        movie_similarity_df[movie_id]
        .sort_values(ascending=False)
    )

    similar_movies = similar_movies.iloc[
        1:n_movies+1
    ]

    similar_movies = similar_movies[
        movie_rating_counts[
            similar_movies.index
        ] >= min_ratings
    ]

    recommended_movies = movies[
        movies["movieId"].isin(
            similar_movies.index
        )
    ][["movieId", "title", "metadata"]].copy()

    recommended_movies["collaborative_score"] = (
        recommended_movies["movieId"]
        .map(similar_movies)
        .round(round)
    )

    recommended_movies = (
        recommended_movies
        .sort_values(
            by="collaborative_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return recommended_movies


def hybrid_recommend(
    title,
    n_movies=10,
    min_ratings=20,
    round=4
):
    movie = find_movie(title)

    if movie is None:
        return "Movie not found."

    title = movie["title"]
    movie_id = movie["movieId"]

    rating_count = movie_rating_counts.get(
        movie_id,
        0
    )

    if rating_count >= 50:
        content_weight = 0.2

    elif rating_count >= 20:
        content_weight = 0.4

    else:
        content_weight = 0.7

    collaborative_weight = 1.0 - content_weight

    content_df = content_recommend(
        title=title,
        n_movies=100,
        round=round
    )

    collaborative_df = collaborative_recommend(
        title=title,
        n_movies=100,
        min_ratings=min_ratings,
        round=round
    )

    if isinstance(content_df, str):
        content_df = pd.DataFrame(
            columns=["movieId", "title", "metadata", "content_score"]
        )

    if isinstance(collaborative_df, str):
        collaborative_df = pd.DataFrame(
            columns=["movieId", "title", "metadata", "collaborative_score"]
        )

    hybrid_df = pd.merge(
        content_df,
        collaborative_df,
        on=["movieId", "title"],
        how="outer"
    )

    hybrid_df["metadata"] = (
        hybrid_df["metadata_x"]
        .fillna(hybrid_df["metadata_y"])
    )

    hybrid_df = hybrid_df.drop(
        columns=["metadata_x", "metadata_y"]
    )

    hybrid_df["content_score"] = (
        hybrid_df["content_score"]
        .fillna(0)
    )

    hybrid_df["collaborative_score"] = (
        hybrid_df["collaborative_score"]
        .fillna(0)
    )

    hybrid_df["hybrid_score"] = (
        hybrid_df["content_score"] * content_weight
        +
        hybrid_df["collaborative_score"] * collaborative_weight
    )

    hybrid_df["hybrid_score"] = (
        hybrid_df["hybrid_score"]
        .round(round)
    )

    hybrid_df = (
        hybrid_df
        .sort_values(
            by="hybrid_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    hybrid_df = hybrid_df[
        [
            "movieId",
            "title",
            "metadata",
            "content_score",
            "collaborative_score",
            "hybrid_score"
        ]
    ]

    return hybrid_df.head(n_movies)


# -------------------------------
# User-based recommender functions
# -------------------------------

def get_user_top_rated_movies(
    user_id,
    n_movies=10
):
    if user_id not in user_movie_matrix.index:
        return "Invalid user ID."

    user_ratings = ratings[
        ratings["userId"] == user_id
    ].copy()

    user_ratings = (
        user_ratings
        .sort_values(
            by=["rating", "timestamp"],
            ascending=[False, False]
        )
        .head(n_movies)
    )

    result = user_ratings.merge(
        movies[
            ["movieId", "title", "metadata"]
        ],
        on="movieId",
        how="left"
    )

    result = result[
        [
            "movieId",
            "title",
            "metadata",
            "rating"
        ]
    ].reset_index(drop=True)

    return result


def get_similar_users(
    user_id,
    n=50
):
    if user_id not in user_similarity_df.columns:
        return "Invalid user ID."

    similar_users = (
        user_similarity_df[user_id]
        .sort_values(ascending=False)
    )

    similar_users = similar_users.iloc[
        1:n+1
    ]

    return similar_users


def get_normalized_similar_users(
    user_id,
    n=50
):
    if user_id not in normalized_similarity_df.columns:
        return "Invalid user ID."

    similar_users = (
        normalized_similarity_df[user_id]
        .sort_values(ascending=False)
    )

    similar_users = similar_users.iloc[
        1:n+1
    ]

    return similar_users


def recommend_movies(
    user_id,
    n_movies=10,
    min_ratings=20,
    n_similar_users=50,
    round=4
):
    if user_id not in user_movie_matrix.index:
        return "Invalid user ID."

    similar_users = get_similar_users(
        user_id,
        n=n_similar_users
    )

    if isinstance(similar_users, str):
        return similar_users

    similar_users_ratings = user_movie_matrix.loc[
        similar_users.index
    ]

    mean_ratings = similar_users_ratings.mean(
        axis=0
    )

    watched_movies = (
        user_movie_matrix.loc[user_id]
        .dropna()
        .index
    )

    recommendations = mean_ratings.drop(
        watched_movies
    )

    recommendations = recommendations[
        movie_rating_counts[
            recommendations.index
        ] >= min_ratings
    ]

    recommendations = (
        recommendations
        .sort_values(ascending=False)
        .head(n_movies)
    )

    recommended_movies = movies[
        movies["movieId"].isin(
            recommendations.index
        )
    ][["movieId", "title", "metadata"]].copy()

    recommended_movies["predicted_rating"] = (
        recommended_movies["movieId"]
        .map(recommendations)
        .round(round)
    )

    recommended_movies = (
        recommended_movies
        .sort_values(
            by="predicted_rating",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return recommended_movies


def weighted_recommend_movies(
    user_id,
    n_movies=10,
    min_ratings=20,
    n_similar_users=50,
    round=4
):
    if user_id not in user_movie_matrix.index:
        return "Invalid user ID."

    similar_users = get_similar_users(
        user_id,
        n=n_similar_users
    )

    if isinstance(similar_users, str):
        return similar_users

    similarity_scores = similar_users.values

    similar_users_ratings = user_movie_matrix.loc[
        similar_users.index
    ]

    weighted_ratings = similar_users_ratings.mul(
        similarity_scores,
        axis=0
    )

    weighted_sum = weighted_ratings.sum(axis=0)

    similarity_sum = (
        similar_users_ratings.notna()
        .mul(similarity_scores, axis=0)
        .sum(axis=0)
    )

    weighted_average = weighted_sum / similarity_sum

    watched_movies = (
        user_movie_matrix.loc[user_id]
        .dropna()
        .index
    )

    recommendations = weighted_average.drop(
        watched_movies
    )

    recommendations = recommendations[
        movie_rating_counts[
            recommendations.index
        ] >= min_ratings
    ]

    recommendations = (
        recommendations
        .sort_values(ascending=False)
        .head(n_movies)
    )

    recommended_movies = movies[
        movies["movieId"].isin(
            recommendations.index
        )
    ][["movieId", "title", "metadata"]].copy()

    recommended_movies["predicted_rating"] = (
        recommended_movies["movieId"]
        .map(recommendations)
        .round(round)
    )

    recommended_movies = (
        recommended_movies
        .sort_values(
            by="predicted_rating",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return recommended_movies


def normalized_weighted_recommend_movies(
    user_id,
    n_movies=10,
    min_ratings=20,
    n_similar_users=50,
    round=4
):
    if user_id not in user_movie_matrix.index:
        return "Invalid user ID."

    similar_users = get_normalized_similar_users(
        user_id,
        n=n_similar_users
    )

    if isinstance(similar_users, str):
        return similar_users

    similarity_scores = similar_users.values

    similar_users_ratings = normalized_matrix.loc[
        similar_users.index
    ]

    weighted_ratings = similar_users_ratings.mul(
        similarity_scores,
        axis=0
    )

    weighted_sum = weighted_ratings.sum(axis=0)

    similarity_sum = (
        similar_users_ratings.notna()
        .mul(abs(similarity_scores), axis=0)
        .sum(axis=0)
    )

    predicted_scores = weighted_sum / similarity_sum

    predicted_scores = (
        predicted_scores
        + user_mean_ratings[user_id]
    )

    predicted_scores = (
        predicted_scores
        / predicted_scores.max()
    ) * 5

    watched_movies = (
        user_movie_matrix.loc[user_id]
        .dropna()
        .index
    )

    recommendations = predicted_scores.drop(
        watched_movies
    )

    recommendations = recommendations[
        movie_rating_counts[
            recommendations.index
        ] >= min_ratings
    ]

    recommendations = (
        recommendations
        .sort_values(ascending=False)
        .head(n_movies)
    )

    recommended_movies = movies[
        movies["movieId"].isin(
            recommendations.index
        )
    ][["movieId", "title", "metadata"]].copy()

    recommended_movies["predicted_rating"] = (
        recommended_movies["movieId"]
        .map(recommendations)
        .round(round)
    )

    recommended_movies = (
        recommended_movies
        .sort_values(
            by="predicted_rating",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return recommended_movies


# -------------------------------
# Sidebar page selection
# -------------------------------

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose page",
    [
        "Movie Recommendation",
        "User-Based Collaborative Analysis"
    ]
)


# -------------------------------
# Page 1: Movie Recommendation
# -------------------------------

if page == "Movie Recommendation":

    st.header("Movie Recommendation")

    method = st.sidebar.selectbox(
        "Recommendation method",
        [
            "Hybrid Filtering",
            "Content-Based Filtering",
            "Collaborative Filtering"
        ]
    )

    n_movies = st.sidebar.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    min_ratings = st.sidebar.slider(
        "Minimum rating count",
        min_value=1,
        max_value=100,
        value=20
    )

    round_value = st.sidebar.slider(
        "Score decimals",
        min_value=2,
        max_value=5,
        value=3
    )

    movie_input = st.text_input(
        "Enter a movie title",
        value="Shrek"
    )

    selected_movie = find_movie(movie_input)

    if selected_movie is not None:
        movie_id = selected_movie["movieId"]

        rating_count = movie_rating_counts.get(
            movie_id,
            0
        )

        st.info(
            f"Matched movie: **{selected_movie['title']}** | "
            f"Rating count: **{rating_count}**"
        )

    else:
        st.warning("No matching movie found yet.")

    if st.button("Recommend"):

        if method == "Content-Based Filtering":
            result = content_recommend(
                title=movie_input,
                n_movies=n_movies,
                round=round_value
            )

        elif method == "Collaborative Filtering":
            result = collaborative_recommend(
                title=movie_input,
                n_movies=n_movies,
                min_ratings=min_ratings,
                round=round_value
            )

        else:
            result = hybrid_recommend(
                title=movie_input,
                n_movies=n_movies,
                min_ratings=min_ratings,
                round=round_value
            )

        if isinstance(result, str):
            st.error(result)

        else:
            st.subheader("Recommendations")
            st.dataframe(
                shorten_metadata(result),
                use_container_width=True
            )


# -------------------------------
# Page 2: User-Based Collaborative Analysis
# -------------------------------

else:

    st.header("User-Based Collaborative Analysis")

    user_ids = sorted(
        ratings["userId"].unique()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_user = st.selectbox(
            "Select user ID",
            user_ids,
            index=0
        )

    with col2:
        n_top_rated = st.slider(
            "Top-rated movies",
            min_value=5,
            max_value=20,
            value=10
        )

    with col3:
        n_recommendations = st.slider(
            "Recommendations",
            min_value=5,
            max_value=20,
            value=10
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        n_similar_users = st.slider(
            "Similar users",
            min_value=10,
            max_value=100,
            value=50,
            step=10
        )

    with col5:
        user_min_ratings = st.slider(
            "Minimum rating count",
            min_value=1,
            max_value=100,
            value=20
        )

    with col6:
        user_round_value = st.slider(
            "Score decimals",
            min_value=2,
            max_value=5,
            value=3
        )

    st.subheader(f"Top Rated Movies of User {selected_user}")

    top_rated_movies = get_user_top_rated_movies(
        selected_user,
        n_movies=n_top_rated
    )

    if isinstance(top_rated_movies, str):
        st.error(top_rated_movies)

    else:
        st.dataframe(
            shorten_metadata(top_rated_movies),
            use_container_width=True
        )

    st.subheader("Different Collaborative Recommendation Results")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Basic Collaborative",
            "Weighted Collaborative",
            "Normalized Weighted Collaborative",
            "Similar Users"
        ]
    )

    with tab1:
        basic_result = recommend_movies(
            selected_user,
            n_movies=n_recommendations,
            min_ratings=user_min_ratings,
            n_similar_users=n_similar_users,
            round=user_round_value
        )

        if isinstance(basic_result, str):
            st.error(basic_result)

        else:
            st.dataframe(
                shorten_metadata(basic_result),
                use_container_width=True
            )

    with tab2:
        weighted_result = weighted_recommend_movies(
            selected_user,
            n_movies=n_recommendations,
            min_ratings=user_min_ratings,
            n_similar_users=n_similar_users,
            round=user_round_value
        )

        if isinstance(weighted_result, str):
            st.error(weighted_result)

        else:
            st.dataframe(
                shorten_metadata(weighted_result),
                use_container_width=True
            )

    with tab3:
        normalized_result = normalized_weighted_recommend_movies(
            selected_user,
            n_movies=n_recommendations,
            min_ratings=user_min_ratings,
            n_similar_users=n_similar_users,
            round=user_round_value
        )

        if isinstance(normalized_result, str):
            st.error(normalized_result)

        else:
            st.dataframe(
                shorten_metadata(normalized_result),
                use_container_width=True
            )

    with tab4:
        similar_users = get_similar_users(
            selected_user,
            n=n_similar_users
        )

        if isinstance(similar_users, str):
            st.error(similar_users)

        else:
            similar_users_df = similar_users.reset_index()
            similar_users_df.columns = [
                "userId",
                "similarity_score"
            ]

            similar_users_df["similarity_score"] = (
                similar_users_df["similarity_score"]
                .round(user_round_value)
            )

            st.dataframe(
                similar_users_df,
                use_container_width=True
            )


# -------------------------------
# Dataset summary
# -------------------------------

with st.expander("Dataset Summary"):

    dataset_stats = pd.DataFrame(
        {
            "Metric": [
                "Movies",
                "Users",
                "Ratings"
            ],
            "Value": [
                movies["movieId"].nunique(),
                ratings["userId"].nunique(),
                len(ratings)
            ]
        }
    )

    st.dataframe(
        dataset_stats,
        use_container_width=True
    )
