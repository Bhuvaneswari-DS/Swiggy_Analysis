# Swiggy_Analysis

🍽️ Swiggy Restaurant Recommender
A personalized restaurant recommendation system built with Streamlit, using KMeans Clustering on cleaned Swiggy data. Users can input their preferences like city, cuisine, rating, and cost to receive curated restaurant suggestions.

🚀 Features
🏙️ City Selector

🍜 Cuisine Filter

⭐ Minimum Rating Slider

💰 Maximum Cost (for 2) Slider

🎯 Smart Clustering-based Recommendations

🧠 ML-Powered Matching using KMeans

⚡ Instant results using Streamlit

🧠 Model & Logic
The app loads a pre-trained KMeans clustering model and encoders from models.pkl.

It takes user input (city, cuisine, rating, cost) and:

One-hot encodes categorical features (city, cuisine).

Combines with numerical features (rating, cost).

Reindexes features to match training columns.

Predicts the cluster using the KMeans model.

Filters restaurants in the predicted cluster that match user preferences.

🧪 Usage
Run the app locally:

bash
Copy
Edit
streamlit run stream.py
You’ll be able to interact with filters in the sidebar and view recommendations instantly.

🗃️ Data
The dataset is derived from Swiggy restaurant listings and contains:

name: Restaurant name

city: City

cuisine: Cuisine type

rating: Average rating

cost: Approximate cost for two

🔐 models.pkl Contents
This file includes:

kmeans: Trained KMeans clustering model.

label_encoder: (If used during encoding) for any label-based preprocessing.

one_hot_encoder: Fitted OneHotEncoder for city and cuisine.

encoded_df: The encoded training data.

trained_columns: Column order used during model training, needed for reindexing.

📸 Example Output
Name	City	Cuisine	Rating	Cost
Spicy Bite	Chennai	South Indian	4.3	500
The Curry House	Chennai	South Indian	4.0	450
