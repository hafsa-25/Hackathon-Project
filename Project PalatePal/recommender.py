import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

ohe = model_data["ohe"]
tfidf = model_data["tfidf"]
features = model_data["features"]
df = model_data["dataframe"]

df = df.fillna("")

df["spice_level"] = pd.to_numeric(df["spice_level"], errors="coerce").fillna(0).astype(int)

df["dish_name"] = df["dish_name"].str.lower()
df["main_ingredients"] = df["main_ingredients"].str.lower()
df["contain_seafood"] = df["contain_seafood"].str.lower()
df["cuisine"] = df["cuisine"].str.lower()
df["dietary_type"] = df["dietary_type"].str.lower()

features = np.nan_to_num(features)

liked_dishes = set()
disliked_dishes = set()

def recommend(cuisine, spice_level, dietary_type, top=5,
              avoid_seafood=False, disliked_ingredient=None):

    copied_df = df.copy()

    cuisine = cuisine.lower()
    dietary_type = dietary_type.lower()
    spice_level = int(spice_level)

    copied_df = copied_df[
        (copied_df["spice_level"] == spice_level) &
        (copied_df["cuisine"] == cuisine) &
        (copied_df["dietary_type"] == dietary_type)
    ]

    if avoid_seafood:
        copied_df = copied_df[copied_df["contain_seafood"] == "no"]

    if disliked_ingredient:
        disliked_list = [i.strip().lower() for i in disliked_ingredient.split(",") if i.strip() != ""]
        for ingredient in disliked_list:
            copied_df = copied_df[~copied_df["main_ingredients"].str.contains(ingredient, na=False)]

    copied_df = copied_df[~copied_df["dish_name"].isin(disliked_dishes)]

    if copied_df.empty:
        return pd.DataFrame()

    filtered_index = copied_df.index
    filtered_features = features[filtered_index]

    user_preference = ohe.transform([[cuisine, dietary_type]])
    user_spice = np.array([[spice_level]])
    user_ingredients = tfidf.transform([""]).toarray()
    user_vector = np.hstack([user_preference, user_spice, user_ingredients])
    user_vector = np.nan_to_num(user_vector)

    similarity = cosine_similarity(user_vector, filtered_features)
    similarity_scores = similarity[0]

    for i, idx in enumerate(filtered_index):
        if df.loc[idx, "dish_name"] in liked_dishes:
            similarity_scores[i] += 0.2

    top = min(top, len(similarity_scores))
    top_relative_indices = similarity_scores.argsort()[-top:][::-1]
    top_indices = filtered_index[top_relative_indices]

    return df.loc[top_indices][["dish_name", "cuisine", "spice_level", "dietary_type", "description"]]

def like_dish(dish_name):
    dish_name = dish_name.lower()
    liked_dishes.add(dish_name)
    disliked_dishes.discard(dish_name)
    return f"❤️ Love to hear that you liked {dish_name}!"

def dislike_dish(dish_name):
    dish_name = dish_name.lower()
    disliked_dishes.add(dish_name)
    liked_dishes.discard(dish_name)
    return f"😢 Sad to hear you disliked {dish_name}. We’ll improve next time!"