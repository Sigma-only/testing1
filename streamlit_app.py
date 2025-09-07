import streamlit as st
import pandas as pd
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
import os
os.environ["STREAMLIT_WATCHDOG"] = "false"


# --- Load Data ---
@st.cache_data
def load_metadata():
    try:
        df = pd.read_csv("gamedata.csv", low_memory=False)
        return df
    except Exception as e:
        st.error(f"Error loading gamedata.csv: {e}")
        return pd.DataFrame()

@st.cache_data
def load_ratings():
    try:
        url = "https://drive.google.com/uc?id=1V_woBuQTiOTxj0OjH0Mx-UhyOY7l14Fg"  # Google Drive file ID
        output = "n_ratings.csv"
        gdown.download(url, output, quiet=False, use_cookies=False)
        df = pd.read_csv(output)
        return df
    except Exception as e:
        st.error(f"Error downloading or loading ratings: {e}")
        return pd.DataFrame()

@st.cache_data
def load_users():
    try:
        df = pd.read_csv("steamuser.csv")
        return df
    except Exception as e:
        st.error(f"Error loading steamuser.csv: {e}")
        return pd.DataFrame()

# --- Content-Based Filtering ---
def combine_features(row, fields_to_include):
    features = []
    if 'description' in fields_to_include:
        features.append(row['short_description'])
        features.append(row['detailed_description'])
        features.append(row['about_the_game'])
    if 'genres' in fields_to_include:
        features.append(row['genres'])
    if 'developer' in fields_to_include:
        features.append(row['developer'])
    if 'publisher' in fields_to_include:
        features.append(row['publisher'])
    if 'platforms' in fields_to_include:
        features.append(row['platforms'])
    if 'required_age' in fields_to_include:
        features.append(str(row['required_age']))
    if 'steamspy_tags' in fields_to_include:
        features.append(row['steamspy_tags'])
    return " ".join(features)

def get_content_based_recommendations(metadata, **filters):
    filtered_metadata = metadata.copy()

    if filters.get("game_name"):
        filtered_metadata = filtered_metadata[filtered_metadata['name'].str.lower() == filters["game_name"].lower()]

    if filters.get("developer"):
        filtered_metadata = filtered_metadata[filtered_metadata['developer'].str.contains(filters["developer"], case=False, na=False)]

    if filters.get("publisher"):
        filtered_metadata = filtered_metadata[filtered_metadata['publisher'].str.contains(filters["publisher"], case=False, na=False)]

    if filters.get("platforms"):
        filtered_metadata = filtered_metadata[filtered_metadata['platforms'].str.contains(filters["platforms"], case=False, na=False)]

    if filters.get("required_age") is not None:
        filtered_metadata = filtered_metadata[filtered_metadata['required_age'] == filters["required_age"]]

    if filters.get("genres"):
        filtered_metadata = filtered_metadata[filtered_metadata['genres'].str.contains(filters["genres"], case=False, na=False)]

    if filters.get("steamspy_tags_input"):
        tags = filters["steamspy_tags_input"].lower().split()
        for tag in tags:
            filtered_metadata = filtered_metadata[filtered_metadata['steamspy_tags'].str.contains(tag, case=False, na=False)]

    if filtered_metadata.empty:
        return []

    # TF-IDF filtering
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(filtered_metadata['combined_features'])
    query = filters.get("description_keywords", "")
    if not query:
        return filtered_metadata['name'].head(20).tolist()

    input_vec = tfidf.transform([query])
    sim_scores = linear_kernel(input_vec, tfidf_matrix).flatten()
    indices = sim_scores.argsort()[-20:][::-1]
    return filtered_metadata['name'].iloc[indices].tolist()


# --- Collaborative Filtering ---
def generate_collaborative_recommendations(selected_game_ratings_list, user_item_matrix, games_df):
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    all_scores = {}
    for selected_game, user_rating in selected_game_ratings_list:
        game_appid = selected_game['appid'].iloc[0]
        game_similarity_scores = item_similarity_df[game_appid]
        weighted_scores = game_similarity_scores * user_rating
        for appid, score in weighted_scores.items():
            all_scores[appid] = all_scores.get(appid, 0) + score

    sorted_scores = pd.Series(all_scores).sort_values(ascending=False)
    rated_ids = [g['appid'].iloc[0] for g, _ in selected_game_ratings_list]
    recommendations = sorted_scores.drop(rated_ids, errors='ignore').head(20)

    return [(games_df.loc[games_df['appid'] == appid, 'name'].values[0], score) for appid, score in recommendations.items()]


# --- MAIN APP ---
def main():
    st.title("🎮 Game Recommendation System")
    st.sidebar.title("Choose Recommendation Mode")
    choice = st.sidebar.radio("Select a method:", ["Content-Based Filtering", "Collaborative Filtering"])

    metadata = load_metadata()
    ratings_df = load_ratings()
    users_df = load_users()

    # Normalize column names
    metadata.columns = metadata.columns.str.strip()
    ratings_df.columns = ratings_df.columns.str.strip()
    users_df.columns = users_df.columns.str.strip()

    # Preprocess metadata
    for col in ['short_description', 'developer', 'publisher', 'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags', 'detailed_description', 'about_the_game']:
        metadata[col] = metadata[col].fillna('')
    fields_to_include = ['description', 'genres', 'developer', 'publisher', 'platforms', 'required_age', 'steamspy_tags']
    metadata['combined_features'] = metadata.apply(lambda row: combine_features(row, fields_to_include), axis=1)

    # Preprocess for collaborative filtering
    merged_df = pd.merge(ratings_df, metadata, on="appid")
    merged_df = pd.merge(merged_df, users_df, on="userID")
    user_item_matrix = merged_df.pivot_table(index="userID", columns="appid", values="rating").fillna(0)
    user_item_sparse_matrix = csr_matrix(user_item_matrix.values)

    if choice == "Content-Based Filtering":
        st.header("🔍 Content-Based Filtering")
        game_name = st.text_input("Game Name")
        description_keywords = st.text_input("Description Keywords")
        developer = st.text_input("Developer")
        publisher = st.text_input("Publisher")
        platforms = st.text_input("Platforms")
        required_age = st.number_input("Required Age", min_value=0, max_value=18, step=1)
        genres = st.text_input("Genres")
        steamspy_tags_input = st.text_input("SteamSpy Tags (separated by spaces)")

        if st.button("Generate Recommendations"):
            recs = get_content_based_recommendations(
                metadata,
                game_name=game_name,
                description_keywords=description_keywords,
                developer=developer,
                publisher=publisher,
                platforms=platforms,
                required_age=required_age if required_age > 0 else None,
                genres=genres,
                steamspy_tags_input=steamspy_tags_input,
            )
            if recs:
                st.subheader("Recommended Games:")
                for i, game in enumerate(recs, 1):
                    st.write(f"{i}. {game}")
            else:
                st.warning("No games found with those filters.")

    elif choice == "Collaborative Filtering":
        st.header("👥 Collaborative Filtering")

        if "selected_game_ratings" not in st.session_state:
            st.session_state.selected_game_ratings = []

        game_name = st.text_input("Enter Game Name to Add")
        user_rating = st.slider("Your Rating", 1, 5, 3)

        if st.button("Add Game"):
            selected_game = metadata[metadata['name'].str.lower() == game_name.lower()]
            if not selected_game.empty:
                st.session_state.selected_game_ratings.append((selected_game, user_rating))
                st.success(f"Added {selected_game['name'].iloc[0]} with rating {user_rating}")
            else:
                st.warning("Game not found. Please check the spelling.")

        if st.button("Generate Recommendations"):
            if st.session_state.selected_game_ratings:
                recs = generate_collaborative_recommendations(
                    st.session_state.selected_game_ratings, user_item_matrix, metadata
                )
                st.subheader("Recommended Games:")
                for i, (game, score) in enumerate(recs, 1):
                    st.write(f"{i}. {game} (Similarity Score: {score:.2f})")
            else:
                st.warning("Please add at least one game and rating first.")


if __name__ == "__main__":
    main()
