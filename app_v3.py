from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# -------------------------------
# Page setup
# -------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

st.title("Movie Recommendation System")
st.caption("APP VERSION: 32M sparse-compatible")

st.write(
    "This app uses content-based filtering, item-based collaborative filtering, "
    "hybrid filtering, and user-based collaborative filtering."
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
# Load data
# -------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    movies = pd.read_csv(
        get_data_path("processed_movies.csv")
    )

    ratings = pd.read_csv(
        get_data_path("ratings.csv"),
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64"
        }
    )

    movies["metadata"] = (
        movies["metadata"]
        .fillna("")
        .astype(str)
    )

    movies["title"] = (
        movies["title"]
        .fillna("")
        .astype(str)
    )

    return movies, ratings


with st.spinner("Loading datasets..."):
    movies, ratings = load_data()


# -------------------------------
# Prepare models
# -------------------------------

@st.cache_resource(show_spinner=False)
def prepare_models(movies, ratings):

    # -------------------------------
    # Content-based variables
    # -------------------------------

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(
        movies["metadata"]
    )

    indices = pd.Series(
        movies.index,
        index=movies["title"]
    ).drop_duplicates()


    # -------------------------------
    # Sparse collaborative variables
    # -------------------------------

    movie_rating_counts = (
        ratings.groupby("movieId")
        .size()
    )

    movie_ids = np.sort(
        ratings["movieId"].unique()
    )

    user_ids = np.sort(
        ratings["userId"].unique()
    )

    movie_id_to_col = pd.Series(
        np.arange(len(movie_ids)),
        index=movie_ids
    )

    user_id_to_row = pd.Series(
        np.arange(len(user_ids)),
        index=user_ids
    )

    movie_col_to_id = pd.Series(
        movie_ids,
        index=np.arange(len(movie_ids))
    )

    user_row_to_id = pd.Series(
        user_ids,
        index=np.arange(len(user_ids))
    )

    user_rows = (
        ratings["userId"]
        .map(user_id_to_row)
        .to_numpy()
    )

    movie_cols = (
        ratings["movieId"]
        .map(movie_id_to_col)
        .to_numpy()
    )

    rating_values = (
        ratings["rating"]
        .astype("float32")
        .to_numpy()
    )

    user_movie_sparse = csr_matrix(
        (
            rating_values,
            (user_rows, movie_cols)
        ),
        shape=(
            len(user_ids),
            len(movie_ids)
        ),
        dtype=np.float32
    )

    user_movie_norm = normalize(
        user_movie_sparse,
        axis=1
    )

    movie_user_sparse = (
        user_movie_sparse.T
        .tocsr()
    )

    movie_user_norm = normalize(
        movie_user_sparse,
        axis=1
    )


    # -------------------------------
    # Normalized user-based variables
    # -------------------------------

    user_mean_ratings = (
        ratings.groupby("userId")["rating"]
        .mean()
    )

    normalized_values = (
        ratings["rating"].astype("float32")
        -
        ratings["userId"]
        .map(user_mean_ratings)
        .astype("float32")
    ).to_numpy()

    normalized_user_movie_sparse = csr_matrix(
        (
            normalized_values,
            (user_rows, movie_cols)
        ),
        shape=(
            len(user_ids),
            len(movie_ids)
        ),
        dtype=np.float32
    )

    normalized_user_movie_norm = normalize(
        normalized_user_movie_sparse,
        axis=1
    )

    return {
        "tfidf": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "indices": indices,
        "movie_rating_counts": movie_rating_counts,
        "movie_ids": movie_ids,
        "user_ids": user_ids,
        "movie_id_to_col": movie_id_to_col,
        "user_id_to_row": user_id_to_row,
        "movie_col_to_id": movie_col_to_id,
        "user_row_to_id": user_row_to_id,
        "user_movie_sparse": user_movie_sparse,
        "user_movie_norm": user_movie_norm,
        "movie_user_sparse": movie_user_sparse,
        "movie_user_norm": movie_user_norm,
        "user_mean_ratings": user_mean_ratings,
        "normalized_user_movie_sparse": normalized_user_movie_sparse,
        "normalized_user_movie_norm": normalized_user_movie_norm,
    }


with st.spinner("Preparing sparse recommendation models..."):
    models = prepare_models(movies, ratings)


tfidf = models["tfidf"]
tfidf_matrix = models["tfidf_matrix"]
indices = models["indices"]

movie_rating_counts = models["movie_rating_counts"]
movie_ids = models["movie_ids"]
user_ids = models["user_ids"]

movie_id_to_col = models["movie_id_to_col"]
user_id_to_row = models["user_id_to_row"]
movie_col_to_id = models["movie_col_to_id"]
user_row_to_id = models["user_row_to_id"]

user_movie_sparse = models["user_movie_sparse"]
user_movie_norm = models["user_movie_norm"]
movie_user_sparse = models["movie_user_sparse"]
movie_user_norm = models["movie_user_norm"]

user_mean_ratings = models["user_mean_ratings"]
normalized_user_movie_sparse = models["normalized_user_movie_sparse"]
normalized_user_movie_norm = models["normalized_user_movie_norm"]


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


def shorten_metadata(df, length=140):

    if isinstance(df, str):
        return df

    output = df.copy()

    if "metadata" in output.columns:
        output["metadata"] = (
            output["metadata"]
            .astype(str)
            .str.slice(0, length)
        )

    return output


def get_content_weight(rating_count):

    if rating_count >= 1000:
        return 0.2

    elif rating_count >= 100:
        return 0.4

    else:
        return 0.7


def get_movie_rating_count(movie_id):
    return int(
        movie_rating_counts.get(
            movie_id,
            0
        )
    )


# -------------------------------
# Content-based filtering
# -------------------------------

def content_recommend(
    title,
    num=10,
    round=4
):

    matches = movies[
        movies["title"]
        .str.lower()
        .str.contains(title.lower(), regex=False)
    ]

    if matches.empty:
        return "Movie not found."

    title = matches.iloc[0]["title"]

    idx = indices[title]

    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    sim_scores = list(
        enumerate(sim_scores)
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:num+1]

    movie_indices = [
        i[0] for i in sim_scores
    ]

    scores = [
        i[1] for i in sim_scores
    ]

    result = movies[
        ["movieId", "title", "metadata"]
    ].iloc[movie_indices].copy()

    result["similarity_score"] = scores

    result["similarity_score"] = (
        result["similarity_score"]
        .round(round)
    )

    result = result.reset_index(drop=True)

    return result


# -------------------------------
# Item-based collaborative filtering
# -------------------------------

def collaborative_recommend(
    title,
    n_movies=10,
    min_ratings=100,
    round=4
):

    movie = find_movie(title)

    if movie is None:
        return "Movie not found."

    movie_id = movie["movieId"]

    if movie_id not in movie_id_to_col.index:
        return "This movie has no rating data."

    movie_col = movie_id_to_col[movie_id]

    similarity_scores = (
        movie_user_norm[movie_col]
        .dot(movie_user_norm.T)
        .toarray()
        .flatten()
    )

    similarity_scores[movie_col] = -1

    sorted_indices = np.argsort(
        similarity_scores
    )[::-1]

    candidate_movie_ids = (
        movie_col_to_id
        .iloc[sorted_indices]
        .values
    )

    candidate_scores = similarity_scores[
        sorted_indices
    ]

    similar_movies = pd.Series(
        candidate_scores,
        index=candidate_movie_ids
    )

    similar_movies = similar_movies[
        similar_movies > 0
    ]

    similar_movies = similar_movies[
        movie_rating_counts
        .reindex(similar_movies.index)
        .fillna(0) >= min_ratings
    ]

    similar_movies = similar_movies.head(
        n_movies
    )

    recommended_movies = movies[
        movies["movieId"].isin(
            similar_movies.index
        )
    ][["movieId", "title", "metadata"]].copy()

    recommended_movies["similarity_score"] = (
        recommended_movies["movieId"]
        .map(similar_movies)
        .round(round)
    )

    recommended_movies = (
        recommended_movies
        .sort_values(
            by="similarity_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return recommended_movies


# -------------------------------
# Hybrid filtering
# -------------------------------

def hybrid_recommend(
    title,
    n_movies=10,
    min_ratings=100,
    candidate_pool=500,
    round=4
):

    movie = find_movie(title)

    if movie is None:
        return "Movie not found.", None

    movie_id = movie["movieId"]
    matched_title = movie["title"]

    rating_count = get_movie_rating_count(
        movie_id
    )

    content_weight = get_content_weight(
        rating_count
    )

    collaborative_weight = 1.0 - content_weight

    content_df = content_recommend(
        matched_title,
        num=candidate_pool,
        round=round
    )

    collaborative_df = collaborative_recommend(
        matched_title,
        n_movies=candidate_pool,
        min_ratings=min_ratings,
        round=round
    )

    if isinstance(content_df, str):
        content_df = pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "metadata",
                "similarity_score"
            ]
        )

    if isinstance(collaborative_df, str):
        collaborative_df = pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "metadata",
                "similarity_score"
            ]
        )

    content_df = content_df.rename(
        columns={
            "similarity_score": "content_score"
        }
    )

    collaborative_df = collaborative_df.rename(
        columns={
            "similarity_score": "collaborative_score"
        }
    )

    hybrid_df = pd.merge(
        content_df,
        collaborative_df,
        on=[
            "movieId",
            "title"
        ],
        how="outer"
    )

    hybrid_df["metadata"] = (
        hybrid_df["metadata_x"]
        .fillna(hybrid_df["metadata_y"])
    )

    hybrid_df = hybrid_df.drop(
        columns=[
            "metadata_x",
            "metadata_y"
        ]
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
    ).round(round)

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

    info = {
        "matched_title": matched_title,
        "rating_count": rating_count,
        "content_weight": content_weight,
        "collaborative_weight": collaborative_weight
    }

    return hybrid_df.head(n_movies), info


# -------------------------------
# User-based collaborative filtering
# -------------------------------

def get_user_top_rated_movies(
    user_id,
    n_movies=10
):

    if user_id not in user_id_to_row.index:
        return "Invalid user ID."

    user_ratings = ratings[
        ratings["userId"] == user_id
    ].copy()

    user_ratings = (
        user_ratings
        .sort_values(
            by=[
                "rating",
                "timestamp"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(n_movies)
    )

    result = user_ratings.merge(
        movies[
            [
                "movieId",
                "title",
                "metadata"
            ]
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

    if user_id not in user_id_to_row.index:
        return "Invalid user ID."

    user_row = user_id_to_row[user_id]

    similarity_scores = (
        user_movie_norm[user_row]
        .dot(user_movie_norm.T)
        .toarray()
        .flatten()
    )

    similarity_scores[user_row] = -1

    sorted_indices = np.argsort(
        similarity_scores
    )[::-1]

    top_indices = sorted_indices[:n]

    similar_user_ids = (
        user_row_to_id
        .iloc[top_indices]
        .values
    )

    similar_scores = similarity_scores[
        top_indices
    ]

    similar_users = pd.Series(
        similar_scores,
        index=similar_user_ids
    )

    return similar_users


def get_normalized_similar_users(
    user_id,
    n=50
):

    if user_id not in user_id_to_row.index:
        return "Invalid user ID."

    user_row = user_id_to_row[user_id]

    similarity_scores = (
        normalized_user_movie_norm[user_row]
        .dot(normalized_user_movie_norm.T)
        .toarray()
        .flatten()
    )

    similarity_scores[user_row] = -1

    sorted_indices = np.argsort(
        similarity_scores
    )[::-1]

    top_indices = sorted_indices[:n]

    similar_user_ids = (
        user_row_to_id
        .iloc[top_indices]
        .values
    )

    similar_scores = similarity_scores[
        top_indices
    ]

    similar_users = pd.Series(
        similar_scores,
        index=similar_user_ids
    )

    return similar_users


def recommend_movies(
    user_id,
    n_movies=10,
    min_ratings=100,
    n_similar_users=200,
    round=4
):

    if user_id not in user_id_to_row.index:
        return "Invalid user ID."

    similar_users = get_similar_users(
        user_id,
        n=n_similar_users
    )

    if isinstance(similar_users, str):
        return similar_users

    similar_user_ids = similar_users.index

    similar_users_ratings = ratings[
        ratings["userId"].isin(
            similar_user_ids
        )
    ]

    mean_ratings = (
        similar_users_ratings
        .groupby("movieId")["rating"]
        .mean()
    )

    watched_movies = ratings[
        ratings["userId"] == user_id
    ]["movieId"].unique()

    recommendations = mean_ratings.drop(
        watched_movies,
        errors="ignore"
    )

    recommendations = recommendations[
        movie_rating_counts
        .reindex(recommendations.index)
        .fillna(0) >= min_ratings
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
    min_ratings=100,
    n_similar_users=200,
    round=4
):

    if user_id not in user_id_to_row.index:
        return "Invalid user ID."

    similar_users = get_similar_users(
        user_id,
        n=n_similar_users
    )

    if isinstance(similar_users, str):
        return similar_users

    similar_users_df = (
        similar_users
        .reset_index()
    )

    similar_users_df.columns = [
        "userId",
        "similarity"
    ]

    similar_users_ratings = ratings[
        ratings["userId"].isin(
            similar_users.index
        )
    ].merge(
        similar_users_df,
        on="userId",
        how="left"
    )

    similar_users_ratings["weighted_ratings"] = (
        similar_users_ratings["rating"]
        *
        similar_users_ratings["similarity"]
    )

    weighted_sum = (
        similar_users_ratings
        .groupby("movieId")["weighted_ratings"]
        .sum()
    )

    similarity_sum = (
        similar_users_ratings
        .groupby("movieId")["similarity"]
        .sum()
    )

    weighted_average = (
        weighted_sum / similarity_sum
    )

    watched_movies = ratings[
        ratings["userId"] == user_id
    ]["movieId"].unique()

    recommendations = weighted_average.drop(
        watched_movies,
        errors="ignore"
    )

    recommendations = recommendations[
        movie_rating_counts
        .reindex(recommendations.index)
        .fillna(0) >= min_ratings
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
    min_ratings=100,
    n_similar_users=200,
    round=4
):

    if user_id not in user_id_to_row.index:
        return "Invalid user ID."

    similar_users = get_normalized_similar_users(
        user_id,
        n=n_similar_users
    )

    if isinstance(similar_users, str):
        return similar_users

    similar_users_df = (
        similar_users
        .reset_index()
    )

    similar_users_df.columns = [
        "userId",
        "similarity"
    ]

    similar_users_ratings = ratings[
        ratings["userId"].isin(
            similar_users.index
        )
    ].merge(
        similar_users_df,
        on="userId",
        how="left"
    )

    similar_users_ratings["user_mean"] = (
        similar_users_ratings["userId"]
        .map(user_mean_ratings)
    )

    similar_users_ratings["normalized_rating"] = (
        similar_users_ratings["rating"]
        -
        similar_users_ratings["user_mean"]
    )

    similar_users_ratings["weighted_normalized_rating"] = (
        similar_users_ratings["normalized_rating"]
        *
        similar_users_ratings["similarity"]
    )

    weighted_sum = (
        similar_users_ratings
        .groupby("movieId")["weighted_normalized_rating"]
        .sum()
    )

    similarity_sum = (
        similar_users_ratings
        .assign(
            abs_similarity=lambda x: x["similarity"].abs()
        )
        .groupby("movieId")["abs_similarity"]
        .sum()
    )

    predicted_scores = (
        weighted_sum / similarity_sum
    )

    predicted_scores = (
        predicted_scores
        +
        user_mean_ratings[user_id]
    )

    predicted_scores = (
        predicted_scores
        /
        predicted_scores.max()
    ) * 5

    watched_movies = ratings[
        ratings["userId"] == user_id
    ]["movieId"].unique()

    recommendations = predicted_scores.drop(
        watched_movies,
        errors="ignore"
    )

    recommendations = recommendations[
        movie_rating_counts
        .reindex(recommendations.index)
        .fillna(0) >= min_ratings
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
# Sidebar navigation
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
            "Item-Based Collaborative Filtering"
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
        max_value=1000,
        value=100
    )

    candidate_pool = st.sidebar.slider(
        "Candidate pool",
        min_value=100,
        max_value=1000,
        value=500,
        step=100
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

    selected_movie = find_movie(
        movie_input
    )

    if selected_movie is not None:
        movie_id = selected_movie["movieId"]
        rating_count = get_movie_rating_count(
            movie_id
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
                num=n_movies,
                round=round_value
            )

            info = None

        elif method == "Item-Based Collaborative Filtering":
            result = collaborative_recommend(
                title=movie_input,
                n_movies=n_movies,
                min_ratings=min_ratings,
                round=round_value
            )

            info = None

        else:
            result, info = hybrid_recommend(
                title=movie_input,
                n_movies=n_movies,
                min_ratings=min_ratings,
                candidate_pool=candidate_pool,
                round=round_value
            )

        if isinstance(result, str):
            st.error(result)

        else:
            if info is not None:
                st.write(
                    f"Rating count: **{info['rating_count']}** | "
                    f"Content weight: **{info['content_weight']}** | "
                    f"Collaborative weight: **{info['collaborative_weight']}**"
                )

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

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_user = st.number_input(
            "Enter user ID",
            min_value=int(user_ids.min()),
            max_value=int(user_ids.max()),
            value=int(user_ids.min()),
            step=1
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
            min_value=50,
            max_value=500,
            value=200,
            step=50
        )

    with col5:
        user_min_ratings = st.slider(
            "Minimum rating count",
            min_value=1,
            max_value=1000,
            value=100
        )

    with col6:
        user_round_value = st.slider(
            "Score decimals",
            min_value=2,
            max_value=5,
            value=3
        )

    if st.button("Analyze User"):

        st.subheader(
            f"Top Rated Movies of User {selected_user}"
        )

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

        st.subheader(
            "Different Collaborative Recommendation Results"
        )

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
                similar_users_df = (
                    similar_users
                    .reset_index()
                )

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
                "Ratings",
                "Movies with ratings"
            ],
            "Value": [
                movies["movieId"].nunique(),
                ratings["userId"].nunique(),
                len(ratings),
                len(movie_ids)
            ]
        }
    )

    st.dataframe(
        dataset_stats,
        use_container_width=True
    )
