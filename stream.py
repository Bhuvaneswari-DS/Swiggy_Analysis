import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load models and encoders
with open("models.pkl", "rb") as f:
    model_data = pickle.load(f)

kmeans = model_data["kmeans"]
label_encoder = model_data["label_encoder"]
one_hot_encoder = model_data["one_hot_encoder"]
encoded_df = model_data["encoded_df"]
trained_columns = model_data["trained_columns"]

# Load cleaned data
df = pd.read_csv("cleaned_swiggy_data.csv")  # Use relative path for portability

# # Streamlit UI
st.set_page_config(page_title="🍽️ Swiggy Restaurant Recommender", layout="wide")
st.title("🍴 Swiggy Restaurant Recommender")
st.markdown("### Get personalized restaurant suggestions based on your preferences.")

# Sidebar filters
st.sidebar.header("🔧 Filter Your Preferences")
city = st.sidebar.selectbox("🏙️ Select City", sorted(df["city"].unique()))
cuisine = st.sidebar.selectbox("🍜 Preferred Cuisine", sorted(df["cuisine"].unique()))
min_rating = st.sidebar.slider("⭐ Minimum Rating", 0.0, 5.0, 3.5, 0.1)
max_cost = st.sidebar.slider("💰 Maximum Price (for 2)", 100, 3000, 800, 50)


# User input DataFrame
input_df = pd.DataFrame([{
    "city": city,
    "cuisine": cuisine,
    "rating": min_rating,
    "cost": max_cost
}])

#Encode city and cuisine
encoded_cat = one_hot_encoder.transform(input_df[["city", "cuisine"]])  # no .toarray()
if hasattr(encoded_cat, "toarray"):  # if it's sparse
    encoded_cat = encoded_cat.toarray()

encoded_cat_df = pd.DataFrame(encoded_cat, columns=one_hot_encoder.get_feature_names_out(["city", "cuisine"]))

# Combine with numerical
numerical_df = input_df[["rating", "cost"]].reset_index(drop=True)
encoded_input = pd.concat([encoded_cat_df.reset_index(drop=True), numerical_df], axis=1)

# Reindex to match training columns
encoded_input = encoded_input.reindex(columns=trained_columns, fill_value=0)

# Predict cluster
cluster = kmeans.predict(encoded_input)[0]

# Filter recommendations
df_encoded = df.copy()
df_encoded["cluster"] = kmeans.labels_



recommendations = df_encoded[
    (df_encoded["cluster"] == cluster) &
    (df_encoded["rating"] >= min_rating) &
    (df_encoded["cost"] <= max_cost) &
    (df_encoded["city"] == city) &
    (df_encoded["cuisine"] == cuisine)
]


# Display
st.subheader("🔍 Recommended Restaurants")
if not recommendations.empty:
    st.dataframe(recommendations[["name", "city", "cuisine", "rating", "cost"]])
else:
    st.warning("No matching restaurants found. Try adjusting your preferences.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")






