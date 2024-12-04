from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import urlparse, urlunparse
from sklearn.metrics.pairwise import linear_kernel


app = Flask(__name__)
popularity_threshold = 5000
df_anime=pd.read_csv('../dataset/anime_cleaned.csv')
df_anime = df_anime.query('members >= @popularity_threshold') # only give rating of those movies for which atleast 600 user have rated
# IMPORTANT! reindexes the dataframe
df_anime = df_anime.drop_duplicates('anime_id').reset_index(drop=True);
# Create a term frequency inverse document frequency
tfidf = TfidfVectorizer(stop_words='english')
# Define a generator to compute TF-IDF matrix on the fly
tfidf_matrix_generator = tfidf.fit_transform((genre for genre in df_anime['genre'].to_numpy().astype('U')))
# Function to get recommendations based on cosine similarity, genre, and ratings based on score
# show_type can be None, Movie, OVA, Special, TV

# Compute cosine similarity matrix as a sparse matrix
cosine_sim_sparse = linear_kernel(tfidf_matrix_generator, tfidf_matrix_generator)

def get_recommendations(title, cosine_sim, df, n=10, show_type=None):
    idx = df[df['title'] == title].index[0]
    print("Genres of title", df.iloc[idx]['genre'])
    # Compute the similarity scores between the anime at the given index and all other animes
    sim_scores = list(enumerate(cosine_sim[idx]))
    #print(sim_scores)
    # Filter out animes with unknown scores
    valid_scores = [x for x in sim_scores if df.iloc[x[0]]['score'] != "UNKNOWN"]
    if show_type:
        valid_scores = [x for x in valid_scores if df.iloc[x[0]]['type'] == show_type]
    
    # Sort the valid anime similarity scores based on the cosine similarity and ratings score in descending order
    sorted_scores = sorted(valid_scores, key=lambda x: (x[1], df.iloc[x[0]]['score']), reverse=True)
    
    # Get the top 10 similar animes (excluding the anime itself)
    top_animes = [x for x in sorted_scores if x[0] != idx][:n]
    #print(top_animes)
    # Extract the indices of the recommended animes
    recommended_indices = [idx for idx, _ in top_animes]
    keys = ['title', 'genre', 'score', 'type', 'image_url', 'anime_id', 'title_english']
    recommended_animes = df.iloc[recommended_indices][keys]
    return zip(recommended_indices,recommended_animes.to_dict(orient='records'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search-complete', methods=['POST'])
def search():
    query = request.form.get('query', '').lower()
    if query:
        # Filter rows where 'title' contains the query (case insensitive)
        matching_titles = df_anime[df_anime['title'].str.contains(query, case=False, na=False) | 
                           df_anime['title_english'].str.contains(query, case=False, na=False) | 
                           df_anime['title_synonyms'].str.contains(query, case=False, na=False)]

        # Get the top 10 results
        matching_titles = matching_titles.head(15)
    else:
        matching_titles = pd.DataFrame()
    #print(matching_titles)
    #_print(matching_titles)
    return render_template('results.html', 
                           results=zip(matching_titles.index, matching_titles.to_dict(orient='records')))

@app.route('/rec/<int:index_id>', methods=['POST'])
def anime_rec(index_id):
    selected= df_anime.iloc[index_id]
    anime=selected.to_dict()
    recs = get_recommendations(selected['title'], cosine_sim_sparse, df_anime, n=5) 
    #_print(anime)
    updated_url = anime['image_url'].replace('https://myanimelist.cdn-dena.com', 'https://cdn.myanimelist.net')
    anime['image_url']=updated_url
    print(anime)
    results = list(recs)
    for i,r in results:
        updated_url=r['image_url'].replace('https://myanimelist.cdn-dena.com', 'https://cdn.myanimelist.net' )
        r['image_url']=updated_url
    return render_template('anime-rec.html',
                           anime=anime,
                           results=results)


if __name__ == '__main__':
    app.run(debug=True)
