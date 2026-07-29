# 🎬 CineBot AI - Intelligent Movie Recommendation Chatbot

An AI-powered movie recommendation chatbot that understands natural language queries and delivers personalized movie suggestions using **Google Gemini**, **Content-Based Filtering**, and **TMDb APIs**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LLM](https://img.shields.io/badge/Google-Gemini-orange)
![API](https://img.shields.io/badge/API-TMDb-green)
![NLP](https://img.shields.io/badge/NLP-Content--Based-red)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 📖 Overview

CineBot AI is an intelligent movie recommendation chatbot that allows users to discover movies through natural language conversations.

Instead of relying on fixed filters, the chatbot uses **Google Gemini** to understand user intent and extracts preferences such as genre, actor, director, rating, release year, and streaming platform. These preferences are processed by a **content-based recommendation engine** that suggests the most relevant movies. The chatbot also integrates with **The Movie Database (TMDb)** to fetch real-time movie information including posters, ratings, overviews, trailers, and OTT availability.

---

## ✨ Features

- 🤖 AI-powered natural language understanding using Google Gemini
- 🎯 Content-based movie recommendation engine
- 🎭 Recommend movies by:
  - Genre
  - Actor
  - Director
  - Similar Movies
  - IMDb Rating
  - Release Year
  - OTT Platform
- 🎬 Fetch movie posters, trailers, ratings, and descriptions using TMDb APIs
- 💬 Interactive Telegram chatbot
- ⚡ Fast and accurate recommendations
- 📝 Logging and exception handling
- 🔍 Semantic query understanding

---

## 🏗️ System Architecture

```text
                 User Query
                      │
                      ▼
        Telegram Chatbot Interface
                      │
                      ▼
      Google Gemini (Intent Parsing)
                      │
      Extract User Preferences
                      │
                      ▼
  Content-Based Recommendation Engine
     (NLP + Cosine Similarity)
                      │
                      ▼
      TMDb API (Movie Information)
                      │
                      ▼
     Personalized Recommendations
```

---

## 🧠 How It Works

### Step 1: Understand User Intent

The chatbot accepts conversational queries such as:

- Recommend sci-fi movies like Interstellar
- Action movies starring Tom Cruise
- Comedy movies available on Netflix
- Psychological thrillers released after 2020

Google Gemini extracts structured information including:

- Genre
- Actor
- Director
- Movie Name
- OTT Platform
- Rating
- Release Year

---

### Step 2: Generate Recommendations

The recommendation engine processes movie metadata including:

- Genres
- Cast
- Director
- Keywords
- Overview

Movies are converted into feature vectors using NLP techniques and **Cosine Similarity** is used to find the most relevant recommendations.

---

### Step 3: Fetch Real-Time Movie Information

The chatbot retrieves additional information from TMDb including:

- Movie Poster
- Overview
- IMDb Rating
- Runtime
- Release Date
- OTT Availability
- Official Trailer

---

## 🛠️ Tech Stack

### Programming Language

- Python

### AI & NLP

- Google Gemini API
- CountVectorizer
- Cosine Similarity
- Natural Language Processing

### Machine Learning

- Scikit-learn

### APIs

- TMDb API
- Telegram Bot API

### Data Processing

- Pandas
- NumPy

### Logging

- Python Logging Module

---

## 📂 Project Structure

```text
Movie_Recommendation_Chatbot/
│
├── bot.py
├── recommender.py
├── gemini_parser.py
├── tmdb_api.py
├── utils.py
├── dataset/
│   └── movies.csv
├── assets/
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/Tyagi-Akshat/Movie_Recommendation_chatbot.git
```

### Navigate to the project

```bash
cd Movie_Recommendation_chatbot
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
TMDB_API_KEY=YOUR_TMDB_API_KEY
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
```

### Run the chatbot

```bash
python bot.py
```

---

## 💬 Example Queries

```text
Recommend sci-fi movies like Interstellar
```

```text
Suggest action movies starring Tom Cruise
```

```text
Comedy movies available on Netflix
```

```text
Recommend psychological thrillers
```

```text
Movies directed by Christopher Nolan
```

```text
Best horror movies after 2020
```

---

## 📈 Key Features

- AI-powered conversational chatbot
- Google Gemini for intent understanding
- Content-based recommendation engine
- Cosine Similarity for personalized recommendations
- Real-time TMDb API integration
- Telegram Bot interface
- Modular and scalable architecture

---

## 🔮 Future Enhancements

- Conversation memory
- Personalized user profiles
- Watchlist management
- Hybrid recommendation system
- Vector Database (FAISS/Chroma)
- RAG-based movie search
- Web application using React
- Docker deployment
- Cloud deployment on AWS/GCP

---

## 📚 Skills Demonstrated

- Large Language Models (LLMs)
- Prompt Engineering
- Recommendation Systems
- Natural Language Processing
- API Integration
- Backend Development
- Python
- Machine Learning
- Software Design
- Conversational AI

---


## 👨‍💻 Author

**Akshat Tyagi**

- GitHub: https://github.com/Tyagi-Akshat

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
