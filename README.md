# 📽️Video Vault: AI-Powered YouTube Intelligence Agent
Video Vault is a multimodal, LLM-powered YouTube Analyzer built to deeply understand and interact with video content. From intelligent transcript summarization to sentiment-aware comment filtering and chatbot-style Q&A, this system brings together cutting-edge NLP, LLMs, and DevOps best practices into one scalable application.

## Features

| Module | Description |
| --- | --- |
| 🔎 Video Search | Find YouTube videos via keyword queries |
| 📄 Transcript Fetching | Pull full video transcripts (via YouTubeTranscriptAPI) |
| 🤖 Chatbot Q&A | Perform sentiment analysis & classify comments
| 🧠 Context-Aware Memory | Chatbot retains past context and adjusts responses |
| 🧩 Relevance Filtering | Categorize comments as: Relevant, Spam, or Irrelevant |
| 🌐 Web UI	 | Gradio-based multi-tab interface |
| 🧰 CI/CD + GCP | Dockerized + GitHub Actions + Google Cloud Run |

## Architecture Overview
```
User ↔ Gradio UI
       ↕
LangChain (Gemini Pro / Gemini Flash)
       ↕
Transcript & Comment Processing
       ↕
NLP Models (DistilBERT, Sentence-BERT)
       ↕
Vector Store (FAISS)
       ↕
YouTube APIs & yt-dlp
       ↕
GCP (Cloud Run, GitHub Actions)
```

## Tech Stack
- **Frontend/UI:** Gradio
- ** LLMs & Chains: **Google Gemini Pro / Gemini Flash, LangChain (memory, tools, prompt chaining)
- **NLP Models:** Sentence Embedding: sentence-transformers/all-MiniLM-L6-v2
- **Sentiment Analysis:** distilbert-base-uncased-finetuned-sst-2-english
- **Vector DB:** FAISS
- **External APIs:** YouTube Data API v3, YouTubeTranscriptAPI, yt-dlp, ScraperAPI
- **DevOps & Infra:** Docker, GitHub Actions, Google Cloud Run, GCP Secrets Manager


## Running Locally
> Requires: Python ≥ 3.10, pip, virtualenv
```
git clone https://github.com/RominaFdo/video-vault.git
cd video-vault
```
```
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```
> Create a .env file in the root directory:
```
GOOGLE_API_KEY=your_gemini_key
YOUTUBE_API_KEY=your_youtube_data_api_key
SCRAPERAPI_KEY=your_scraperapi_key
```

> Run 
```
python app.py
```

