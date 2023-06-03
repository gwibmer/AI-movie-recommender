from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__)

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the file paths
movies_file = os.path.join(current_directory, 'movies.csv')
ratings_file = os.path.join(current_directory, 'ratings.csv')

# Read the CSV files
movies = pd.read_csv(movies_file)
ratings = pd.read_csv(ratings_file)


def clean_title(title):
    # Function to remove special characters from the title
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title


# Apply the clean_title function to the "title" column in the movies DataFrame
movies["clean_title"] = movies["title"].apply(clean_title)

# Create a TF-IDF vectorizer to convert movie titles into numerical representations
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])


@app.route('/')
def index():
    # Render the recommendation template
    return recommend()


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        # Retrieve the movie title from the form submission
        title = request.form['title']
        # Clean the title by removing special characters
        title = clean_title(title)
        # Convert the cleaned title into a TF-IDF vector
        query_vec = vectorizer.transform([title])
        # Calculate the similarity between the query vector and all movie vectors
        similarity = cosine_similarity(query_vec, tfidf).flatten()
        # Find the indices of the most similar movies
        indices = np.argpartition(similarity, -5)[-5:]
        # Retrieve the movie details for the most similar movies
        results = movies.iloc[indices].iloc[::-1]
        # Get the movieId of the first movie in the results
        movie_id = results.iloc[0]["movieId"]
        # Find similar movies based on the movieId
        recs = find_similar_movies(movie_id)

        # Convert recommendations to a list of dictionaries
        recs_list = recs.to_dict('records')

        # Render the recommend.html template and pass the recommendations
        return render_template('recommend.html', recommendations=recs_list)
    else:
        # Render the recommend.html template
        return render_template('recommend.html')


def find_similar_movies(movie_id):
    # Find users who rated the given movie highly
    similar_users = ratings[(ratings["movieId"] == movie_id) & (
        ratings["rating"] > 4)]["userId"].unique()
    # Find movies that were rated highly by similar users
    similar_user_recs = ratings[(ratings["userId"].isin(
        similar_users)) & (ratings["rating"] > 4)]["movieId"]
    # Calculate the percentage of similar user recommendations for each movie
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    # Keep only movies with a recommendation percentage higher than 10%
    similar_user_recs = similar_user_recs[similar_user_recs > .10]

    # Find all users who rated the recommended movies highly
    all_users = ratings[(ratings["movieId"].isin(
        similar_user_recs.index)) & (ratings["rating"] > 4)]
    # Calculate the percentage of all user recommendations for each movie
    all_user_recs = all_users["movieId"].value_counts(
    ) / len(all_users["userId"].unique())

    # Concatenate the similar user recommendations and all user recommendations
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    # Calculate a score by dividing the number of similar user recommendations by the number of all user recommendations
    rec_percentages["score"] = rec_percentages["similar"] / \
        rec_percentages["all"]
    # Sort the recommendations by score in descending order
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    # Retrieve the top 10 recommendations and merge with the movies DataFrame to get additional movie details
    recommendations = rec_percentages.head(10).merge(
        movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]
    # Round the score to two decimal places
    recommendations["score"] = recommendations["score"].round(2)
    # Extract only the first three genres
    recommendations["genres"] = recommendations["genres"].apply(
        lambda x: ', '.join(x.split('|')[:3]))
    return recommendations


if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)
