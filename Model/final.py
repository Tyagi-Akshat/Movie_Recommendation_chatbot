import requests
import google.generativeai as genai
import json
from difflib import get_close_matches
import re
import random
import numpy as np
import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration (REPLACE WITH YOUR ACTUAL KEYS AND PREFERENCES) ---
API_KEY = 'Your Tmdb Api Key' # Your TMDb API Key
BASE_URL = 'https://api.themoviedb.org/3'

# Configure the Gemini API - REPLACE WITH YOUR ACTUAL GEMINI API KEY
genai.configure(api_key="Your Gemini Api key")

# --- Global Variables for Context and Phrases ---
# For Engaging Responses
INTRO_PHRASES = [
    "Alright, let's find something great for you!",
    "Searching the cinematic universe now...",
    "One moment while I fetch some awesome recommendations!",
    "Got it! Let's see what we can find...",
    "Excellent choice! Preparing a list of movies for you.",
]

NO_RESULTS_PHRASES = [
    "Hmm, I couldn't find any recommendations matching that specific criteria.",
    "My apologies, no movies found with those exact details.",
    "Looks like that combination is a bit too specific. Maybe try broadening your search?",
    "Sorry, I drew a blank on that one. Can I help with something else?",
]

FINAL_RECS_PHRASES = [
    "Here are some top picks for you!",
    "Enjoy these recommendations!",
    "Hope you find something you love here:",
    "Your movie magic is ready:",
]

# Global state for contextual memory and pagination
last_parsed_query_for_context = {}
last_page_number = 1

# --- TMDb API Helper Functions and Class ---

# Assuming genre_mapping and OTT_MAP are defined elsewhere or fetched
genre_mapping = {
    "action": 28, "adventure": 12, "animation": 16, "comedy": 35,
    "crime": 80, "documentary": 99, "drama": 18, "family": 10751,
    "fantasy": 14, "history": 36, "horror": 27, "music": 10402,
    "mystery": 9648, "romance": 10749, "science fiction": 878,
    "sci-fi": 878, "tv movie": 10770, "thriller": 53, "war": 10752,
    "western": 37
}

OTT_MAP = {
    "netflix": 8, "amazon prime": 9, "disney plus": 337,
    "hulu": 15, "hbo max": 188, "apple tv plus": 350,
    "paramount plus": 531, "peacock": 386, "youtube premium": 167
}

def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {'api_key': API_KEY}
    response = requests.get(url, params=params)
    return response.json()

def get_watch_providers(movie_id, region="US"):
    url = f"{BASE_URL}/movie/{movie_id}/watch/providers"
    params = {'api_key': API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    providers = []
    if region in data.get("results", {}):
        region_data = data["results"][region]
        for service_type in ["flatrate", "buy", "rent"]:
            if service_type in region_data:
                for provider in region_data[service_type]:
                    providers.append(provider["provider_name"])
    return list(set(providers)) # Return unique provider names


class TMDbAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"

    def search_movie_results(self, title):
        url = f"{self.base_url}/search/movie"
        params = {
            "api_key": self.api_key,
            "query": title
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"⚠️ TMDb search error {response.status_code} for '{title}'")
            return []
        data = response.json()
        return data.get("results", [])

    def search_movie(self, title):
        results = self.search_movie_results(title)
        if results:
            return results[0] # Return the whole movie object, not just ID
        else:
            return None

    def search_person(self, name):
        url = f"{self.base_url}/search/person"
        params = {
            "api_key": self.api_key,
            "query": name
        }
        response = requests.get(url, params=params)
        results = response.json().get("results", [])
        return results # Return all results for disambiguation

    def search_movie_disambiguate(self, title): # This method is used for disambiguation purposes in some parts of the code
        return self.search_movie_results(title) # Reuse search_movie_results for disambiguation


    def discover_movies(self, genre=None, actor_id=None, director_id=None, ott=None,
                        year=None, decade=None, min_rating=None, sort_by="popularity.desc", page=1): # Added page parameter
        url = f"{self.base_url}/discover/movie"
        params = {
            "api_key": self.api_key,
            "sort_by": sort_by,
            "vote_count.gte": 50, # Minimum vote count to avoid obscure movies
            "include_adult": False, # Good practice
            "page": page # Add page to parameters
        }

        if genre:
            params["with_genres"] = genre
        if actor_id:
            params["with_cast"] = actor_id
        if director_id:
            params["with_crew"] = director_id # For directors, use 'with_crew'

        if ott:
            params["with_watch_providers"] = ott
            params["watch_region"] = "US" # Or make this configurable
            params["flatrate_execution"] = "flatrate" # For streaming services

        if year:
            params["primary_release_year"] = year

        if decade:
            start_year = int(decade.replace('s', ''))
            end_year = start_year + 9
            params["primary_release_date.gte"] = f"{start_year}-01-01"
            params["primary_release_date.lte"] = f"{end_year}-12-31"

        if min_rating:
            params["vote_average.gte"] = min_rating

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"⚠️ TMDb discover error {response.status_code}: {response.text}")
            return []
        data = response.json()
        return data.get("results", [])

# Initialize TMDbAPI instance
tmdb_api = TMDbAPI(api_key=API_KEY)


# --- Movie Recommendation Engine (using your provided code) ---

# Read data
# Using error_bad_lines=False and warn_bad_lines=True which might help skip problematic lines
# Make sure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the same directory as this script
try:
    movies_df = pd.read_csv(r'C:\Users\Akshat\Desktop\Akshat\Movie_bot\tmdb_5000_movies.csv', encoding='latin-1', on_bad_lines='skip')
    credits_df = pd.read_csv(r'C:\Users\Akshat\Desktop\Akshat\Movie_bot\tmdb_5000_credits.csv', encoding='latin-1', on_bad_lines='skip', quotechar='"')
except FileNotFoundError:
    print("Error: Make sure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the same directory.")
    print("Movie recommendation (similarity) feature will be limited without these files.")
    movies_df = pd.DataFrame() # Create empty DataFrame to avoid errors later
    credits_df = pd.DataFrame()

# Conditional processing only if dataframes are not empty
if not movies_df.empty and not credits_df.empty:
    movies_df = movies_df.merge(credits_df, on='title')

    # Select relevant columns
    movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    # Drop nulls
    movies_df.dropna(inplace=True)

    # Convert stringified JSON to list of names
    def convert(obj):
        L = []
        try: # Add try-except for robust parsing of stringified JSON
            for i in ast.literal_eval(obj):
                L.append(i['name'])
        except (ValueError, SyntaxError):
            pass # Handle cases where obj is not valid JSON string
        return L

    # Extract director name
    def fetch_director(obj):
        L = []
        try:
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    L.append(i['name'])
                    break
        except (ValueError, SyntaxError):
            pass
        return L

    # Get top 3 cast members
    def convert3(obj):
        L = []
        counter = 0
        try:
            for i in ast.literal_eval(obj):
                if counter < 3: # Changed from != 3 to < 3 for clarity and correctness
                    L.append(i['name'])
                    counter += 1
                else:
                    break
        except (ValueError, SyntaxError):
            pass
        return L

    # Apply transformations
    movies_df['genres'] = movies_df['genres'].apply(convert)
    movies_df['keywords'] = movies_df['keywords'].apply(convert)
    movies_df['cast'] = movies_df['cast'].apply(convert3)
    movies_df['crew'] = movies_df['crew'].apply(fetch_director)
    movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())

    # Remove spaces in tags
    movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

    # Create combined tags
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']

    # New dataframe
    new_df = movies_df[['movie_id', 'title', 'tags']].copy() # Use .copy() to avoid SettingWithCopyWarning
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

    # Stemming
    ps = PorterStemmer()
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)

    new_df['tags'] = new_df['tags'].apply(stem)

    # Vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    # Similarity matrix
    similarity = cosine_similarity(vectors)

    # Recommend function
    def recommend(movie):
        movie_lower = movie.lower()
        # Use .str.contains with regex=False for simple substring matching
        matches = new_df[new_df['title'].str.lower().str.contains(movie_lower, na=False, regex=False)]

        if matches.empty:
            print(f"⚠️ No movie found matching '{movie}' in your dataset for similarity.")
            return []

        # If multiple matches, pick the first one for simplicity for now
        # For a more robust solution, you might want to disambiguate here too
        movie_index = matches.index[0]

        distances = similarity[movie_index]
        movies_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:6] # Top 5 similar movies, excluding itself
        return [new_df.iloc[i[0]].title for i in movies_list]
else:
    # Handle case where movie data couldn't be loaded, define a placeholder recommend function
    def recommend(movie):
        print("⚠️ Movie recommendation (similarity) feature is unavailable due to missing data files.")
        return []


# --- Gemini Parsing Function ---
def parse_query_with_gemini(user_input):
    model = genai.GenerativeModel('gemini-1.5-pro-latest') # Or 'gemini-2.5-pro' if available and preferred

    prompt = f"""
    Analyze the following movie recommendation request and extract the following information.
    Provide the output in a JSON format ONLY. Do NOT include any other text, explanations, or markdown wrappers outside the JSON object.
    If a piece of information is not present or cannot be determined, omit it from the JSON.

    Required fields (if applicable):
    - "intent": "discover" for finding new movies based on criteria, "similar" for finding movies like a given title.
    - "title": The title of a movie for 'similar' intent.
    - "genre": The general genre of movies (e.g., "action", "comedy", "drama").
    - "actor": The name of an actor.
    - "director": The name of a director.
    - "ott": The name of an OTT platform (e.g., "Netflix", "Amazon Prime", "Disney Plus").
    - "year": A specific release year (e.g., 2020).
    - "decade": A decade (e.g., "90s", "2000s").
    - "min_rating": A minimum average rating (e.g., 7.5).

    Here are the known genres for reference: {list(genre_mapping.keys())}.
    Here are the known OTT platforms for reference: {list(OTT_MAP.keys())}.

    User request: "{user_input}"

    Example outputs (JSON ONLY):
    {{"intent": "discover", "genre": "action"}}
    {{"intent": "similar", "title": "Inception"}}
    {{"intent": "discover", "actor": "Brad Pitt"}}
    {{"intent": "discover", "director": "Christopher Nolan", "ott": "Netflix"}}
    {{"intent": "discover", "genre": "comedy", "ott": "hulu", "actor": "Adam Sandler"}}
    {{"intent": "discover", "genre": "sci-fi"}}
    {{"intent": "discover", "director": "Quentin Tarantino"}}
    {{"intent": "discover", "genre": "drama", "year": 2023}}
    {{"intent": "discover", "decade": "80s", "min_rating": 7.0}}
    {{"intent": "discover", "genre": "animation", "ott": "disney plus", "year": 2024}}
    """

    try:
        response = model.generate_content(prompt)
        text_content = response.text.strip()

        # Debug print to see raw Gemini output
        print(f"--- DEBUG: Raw Gemini response text ---")
        print(text_content)
        print(f"------------------------------------")

        # Robustly extract JSON using regex for the content between the first { and last }
        json_match = re.search(r'\{.*\}', text_content, re.DOTALL)

        if json_match:
            json_string = json_match.group(0)
            parsed_data = json.loads(json_string)
            return parsed_data
        else:
            # If no JSON object is found by regex, raise an error to trigger fallback
            raise json.JSONDecodeError("No JSON object found in Gemini response.", text_content, 0)

    except json.JSONDecodeError as e:
        print(f"⚠️ Gemini response was not valid JSON: {e}. Raw content attempted to parse: '{text_content}'. Falling back to simple parsing.")
        # Fallback logic
        cleaned = user_input.strip().lower()
        parsed = {
            "intent": None, "title": None, "genre": None, "actor": None, "director": None, "ott": None,
            "year": None, "decade": None, "min_rating": None # Include new fields in fallback
        }
        genre_candidate = get_close_matches(cleaned, genre_mapping.keys(), n=1, cutoff=0.7)
        if genre_candidate:
            parsed["intent"] = "discover"
            parsed["genre"] = genre_mapping[genre_candidate[0]]
        return parsed
    except Exception as e:
        print(f"⚠️ An unexpected error occurred with Gemini API: {e}. Falling back to simple parsing.")
        # Fallback logic
        cleaned = user_input.strip().lower()
        parsed = {
            "intent": None, "title": None, "genre": None, "actor": None, "director": None, "ott": None,
            "year": None, "decade": None, "min_rating": None # Include new fields in fallback
        }
        genre_candidate = get_close_matches(cleaned, genre_mapping.keys(), n=1, cutoff=0.7)
        if genre_candidate:
            parsed["intent"] = "discover"
            parsed["genre"] = genre_mapping[genre_candidate[0]]
        return parsed


# --- Recommendation Orchestrator ---
def get_recommendations(parsed, num_results=5):
    results = []

    if parsed.get("intent") == "similar":
        title_for_similarity = parsed.get("title")
        if not title_for_similarity:
            print("To find similar movies, please specify a movie title.")
            return []

        similar_titles = recommend(title_for_similarity)

        if not similar_titles:
            print(f"⚠️ No similar movies found for '{title_for_similarity}' in the dataset. Showing top-rated movies instead.")
            # Fallback to general top-rated if similarity fails
            top_movies = tmdb_api.discover_movies(sort_by="vote_average.desc")
            if not top_movies:
                return []
            for m in top_movies[:num_results]:
                movie_id = m["id"]
                providers = get_watch_providers(movie_id)
                results.append({
                    "title": m["title"],
                    "overview": m.get("overview", "No description."),
                    "providers": providers
                })
            return results

        for title in similar_titles[:num_results]:
            # Corrected call: tmdb_api.search_movie is now a method
            tmdb_data_search_result = tmdb_api.search_movie(title)
            if not tmdb_data_search_result:
                continue
            movie_id = tmdb_data_search_result["id"]
            movie_details = get_movie_details(movie_id)
            providers = get_watch_providers(movie_id)
            overview = movie_details.get("overview", "No description.")
            results.append({
                "title": movie_details["title"],
                "overview": overview,
                "providers": providers
            })
        return results

    # DISCOVER intent
    actor_id = None
    director_id = None

    # Handle actor disambiguation
    if parsed.get("actor"):
        people = tmdb_api.search_person(parsed["actor"])
        if len(people) > 1:
            print(f"\n❓ There are multiple people named '{parsed['actor']}'. Which one did you mean?")
            for i, p in enumerate(people[:5]): # Show up to 5 options
                known_for = p.get('known_for_department', 'Unknown')
                print(f"   {i+1}. {p['name']} ({known_for})")
            while True:
                try:
                    choice = input("Enter number (or 'skip' to ignore): ")
                    if choice.lower() == 'skip':
                        break
                    choice = int(choice) - 1
                    if 0 <= choice < len(people[:5]):
                        actor_id = people[choice]["id"]
                        parsed["actor_resolved_name"] = people[choice]["name"] # Store resolved name for context
                        break
                    else:
                        print("Invalid choice. Please enter a number from the list or 'skip'.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'skip'.")
        elif people:
            actor_id = people[0]["id"]
            parsed["actor_resolved_name"] = people[0]["name"]
        else:
            print(f"Couldn't find an actor named '{parsed['actor']}'.")


    # Handle director disambiguation (similar logic)
    if parsed.get("director"):
        people = tmdb_api.search_person(parsed["director"])
        if len(people) > 1:
            print(f"\n❓ There are multiple people named '{parsed['director']}'. Which one did you mean?")
            for i, p in enumerate(people[:5]):
                known_for = p.get('known_for_department', 'Unknown')
                print(f"   {i+1}. {p['name']} ({known_for})")
            while True:
                try:
                    choice = input("Enter number (or 'skip' to ignore): ")
                    if choice.lower() == 'skip':
                        break
                    choice = int(choice) - 1
                    if 0 <= choice < len(people[:5]):
                        director_id = people[choice]["id"]
                        parsed["director_resolved_name"] = people[choice]["name"]
                        break
                    else:
                        print("Invalid choice. Please enter a number from the list or 'skip'.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'skip'.")
        elif people:
            director_id = people[0]["id"]
            parsed["director_resolved_name"] = people[0]["name"]
        else:
            print(f"Couldn't find a director named '{parsed['director']}'.")

    genre_id = None
    if parsed.get("genre"):
        genre_id = genre_mapping.get(parsed["genre"].lower())

    ott_id = None
    if parsed.get("ott"):
        ott_id = OTT_MAP.get(parsed["ott"].lower())

    year = parsed.get("year")
    decade = parsed.get("decade")
    min_rating = parsed.get("min_rating")

    current_page = parsed.get('page', 1) # Get page from parsed, default to 1


    discover = tmdb_api.discover_movies(
        genre=genre_id,
        actor_id=actor_id,
        director_id=director_id,
        ott=ott_id,
        year=year,
        decade=decade,
        min_rating=min_rating,
        sort_by="vote_average.desc", # Using vote_average.desc as default for quality
        page=current_page # Pass the page parameter
    )
    if not discover:
        print(random.choice(NO_RESULTS_PHRASES)) # Engaging no results from TMDb
        return []

    for m in discover[:num_results]:
        movie_id = m["id"]
        providers = get_watch_providers(movie_id)
        results.append({
            "title": m["title"],
            "overview": m.get("overview", "No description."),
            "providers": providers
        })

    return results


# --- Main Chat Loop ---
def chat_loop():
    print("🎬 Welcome to MovieBot!")
    print("💡 You can type queries like:")
    print("   - Recommend action movies")
    print("   - Show me movies like Inception")
    print("   - Find movies with Brad Pitt")
    print("   - Christopher Nolan movies on Netflix")
    print("   - Comedy films from the 90s on Hulu with Adam Sandler")
    print("   - Highly-rated sci-fi movies from 2023")
    print("   - Try asking 'more' after a search to get next results!")
    print("Type 'exit' to quit.\n")

    global last_parsed_query_for_context, last_page_number
    last_parsed_query_for_context = {} # Reset for new session
    last_page_number = 1 # Reset for new session

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Hope you found something to watch!")
            break

        parsed_from_gemini = parse_query_with_gemini(user_input)

        # This will be the dictionary of parameters passed to get_recommendations
        current_request_params = parsed_from_gemini.copy()

        # Prepare simplified versions for comparison (excluding 'page' and resolved names)
        keys_for_comparison = ["intent", "genre", "actor", "director", "ott", "year", "decade", "min_rating", "title"]
        current_core_filters = {k: current_request_params.get(k) for k in keys_for_comparison if k in current_request_params}
        last_core_filters = {k: last_parsed_query_for_context.get(k) for k in keys_for_comparison if k in last_parsed_query_for_context}

        # Check if the current request is essentially the same as the last *core* request
        # AND it's a discover intent that can be paginated.
        is_same_discover_query = (
            current_core_filters == last_core_filters and
            current_request_params.get("intent") == "discover"
        )

        # Check if the user is explicitly asking for "more" or "next" results, without changing filters
        is_generic_more_request_phrase = (
            "more" in user_input.lower() or "next" in user_input.lower() or "another" in user_input.lower()
        )

        # Logic for managing pagination and context
        if is_generic_more_request_phrase:
            # User wants more of the *previous* results (if context exists and was discover intent)
            if last_parsed_query_for_context.get("intent") == "discover":
                current_request_params.update(last_parsed_query_for_context) # Inherit all last filters
                last_page_number += 1
                current_request_params['page'] = last_page_number
            else:
                # Can't get "more" if previous intent wasn't discover or if no context
                last_page_number = 1 # Reset
                current_request_params['page'] = 1
                print("Can't get more results without a previous specific query.")
        elif is_same_discover_query:
            # User repeated the same discover query (e.g., "action movies" again), increment page
            last_page_number += 1
            current_request_params['page'] = last_page_number
        else:
            # This is a new or significantly different query, or not a discover intent, reset page
            last_page_number = 1
            current_request_params['page'] = 1

        # Before sending to get_recommendations, ensure all needed context from last_parsed_query_for_context
        # is merged if the current_request_params came in empty (e.g., for "more" or just "comedy")
        # This handles cases where Gemini might return minimal info for follow-up
        if current_request_params.get("intent") == "discover":
            for key in ["genre", "actor", "director", "ott", "year", "decade", "min_rating"]:
                if key not in current_request_params and last_parsed_query_for_context.get(key):
                    current_request_params[key] = last_parsed_query_for_context[key]
            # Ensure intent is correctly set if it was missing but context has it
            if "intent" not in current_request_params and last_parsed_query_for_context.get("intent"):
                current_request_params["intent"] = last_parsed_query_for_context["intent"]

        if "intent" not in current_request_params or not current_request_params["intent"]:
            print(random.choice(NO_RESULTS_PHRASES))
            continue

        print(f"\n🤔 {random.choice(INTRO_PHRASES)}")
        print("Parsed query (sent to get_recommendations):", current_request_params)

        recs = get_recommendations(current_request_params) # Pass the potentially modified parsed data

        # After get_recommendations (and potential disambiguation), update the global context
        if current_request_params.get("intent") == "discover":
            # Store the *actual* parameters used, including resolved names and the page number
            last_parsed_query_for_context = current_request_params.copy()
            # Clean up temporary keys that are not core query filters
            last_parsed_query_for_context.pop("actor_resolved_name", None)
            last_parsed_query_for_context.pop("director_resolved_name", None)
        else:
            # If it's not a discover intent, clear context for future pagination
            last_parsed_query_for_context = {}
            last_page_number = 1

        if not recs:
            # get_recommendations already prints NO_RESULTS_PHRASES if no recs found from TMDb
            continue

        print(f"\n🍿 {random.choice(FINAL_RECS_PHRASES)}\n") # Engaging final recs
        for idx, r in enumerate(recs, start=1):
            print(f"{idx}. 🎬 {r['title']}")
            print(f"   📄 Plot: {r['overview'][:200]}...")
            if r["providers"]:
                print("   📺 Available on:", ", ".join(r["providers"]))
            else:
                print("   📺 No streaming info.")
        print()

# --- Run the Chatbot ---
if __name__ == "__main__":
    chat_loop()
