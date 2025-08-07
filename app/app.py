import gradio as gr
import yt_dlp
from urllib.parse import urlparse, parse_qs
import json
import subprocess
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from youtube_transcript_api.formatters import TextFormatter
from googleapiclient.discovery import build
from transformers import pipeline
import google.generativeai as genai
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY")  

if not GOOGLE_API_KEY or not YOUTUBE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY and YOUTUBE_API_KEY in your .env file")

if not SCRAPERAPI_KEY:
    print("Warning: SCRAPERAPI_KEY not set. Transcript fetching may fail due to IP blocking.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
API_KEY = YOUTUBE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

global_qa_chain = None
global_memory = None
current_video_title = ""

try:
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Model loading error: {e}")
    sentiment_pipe = None
    model = None

def create_session_with_retry():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# def fetch_transcript_with_scraperapi(video_id):
#     """Fetch transcript using ScraperAPI as proxy"""
#     if not SCRAPERAPI_KEY:
#         return "Error: ScraperAPI key not configured. Please set SCRAPERAPI_KEY environment variable."
    
#     if not video_id:
#         return "Error: Invalid video ID provided."
    
#     try:
#         # ScraperAPI endpoint
#         scraperapi_url = "http://api.scraperapi.com"
        
#         # YouTube transcript API endpoint (we'll scrape the transcript page)
#         youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
#         params = {
#             'api_key': SCRAPERAPI_KEY,
#             'url': youtube_url,
#             'render': 'false',  # Set to true if you need JavaScript rendering
#             'country_code': 'us'  # Use US proxy
#         }
        
#         session = create_session_with_retry()
        
#         # First, let's try the direct transcript API approach with proxy
#         try:
#             # Use ScraperAPI as HTTP proxy for youtube-transcript-api
#             proxies = {
#                 'http': f'http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001',
#                 'https': f'http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001'
#             }
            
#             # Try with the original youtube-transcript-api but through proxy
#             ytt_api = YouTubeTranscriptApi()
#             transcript = ytt_api.fetch(video_id, proxies=proxies)
#             # transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies)
#             if transcript:
#                 formatter = TextFormatter()
#                 result = formatter.format_transcript(transcript)
#                 return result if result else "Error: Empty transcript from direct proxy method"
#             else:
#                 return "Error: No transcript data from direct proxy method"
            
#         except Exception as proxy_error:
#             print(f"Direct proxy approach failed: {proxy_error}")
            
#             # Fallback: Scrape transcript data from YouTube page
#             try:
#                 response = session.get(scraperapi_url, params=params, timeout=30)
                
#                 if response.status_code == 200:
#                     # Extract transcript from the scraped HTML
#                     html_content = response.text
                    
#                     if not html_content:
#                         return "Error: Empty response from ScraperAPI"
                    
#                     # Look for transcript data in the page
#                     transcript_pattern = r'"transcriptRenderer".*?"runs":\s*(\[.*?\])'
#                     match = re.search(transcript_pattern, html_content, re.DOTALL)
                    
#                     if match:
#                         try:
#                             runs_data = json.loads(match.group(1))
#                             transcript_text = ""
                            
#                             for run in runs_data:
#                                 if 'text' in run:
#                                     transcript_text += run['text'] + " "
                            
#                             if transcript_text.strip():
#                                 return transcript_text.strip()
#                             else:
#                                 return "Error: No transcript text found in scraped data"
                                
#                         except json.JSONDecodeError as json_error:
#                             return f"Error: Failed to parse transcript data - {str(json_error)}"
#                     else:
#                         # Try alternative patterns or methods
#                         return fetch_transcript_alternative_method(html_content, video_id)
#                 else:
#                     return f"Error: ScraperAPI request failed with status code: {response.status_code}"
                    
#             except Exception as scrape_error:
#                 return f"Error: Failed to scrape with ScraperAPI - {str(scrape_error)}"
                
#     except Exception as e:
#         return f"Error: Exception in ScraperAPI transcript fetch - {str(e)}"
def fetch_transcript_with_scraperapi(video_id):
    """
    Fetches a YouTube transcript by using ScraperAPI to proxy a request
    to the video's page and then scrapes the transcript data from the HTML.

    This method avoids the `youtube-transcript-api` library, which does not
    natively support proxies, thus solving the original error.
    """
    if not SCRAPERAPI_KEY:
        return "Error: ScraperAPI key not configured. Please set SCRAPERAPI_KEY environment variable."
    
    if not video_id:
        return "Error: Invalid video ID provided."
    
    try:
        # ScraperAPI endpoint configuration
        scraperapi_url = "http://api.scraperapi.com"
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
        params = {
            'api_key': SCRAPERAPI_KEY,
            'url': youtube_url,
            'render': 'false', # No need for full rendering to find transcript
            'country_code': 'us'
        }
        
        session = create_session_with_retry()
        
        print(f"Fetching YouTube page for video ID: {video_id} using ScraperAPI...")
        response = session.get(scraperapi_url, params=params, timeout=30)
        
        if response.status_code == 200:
            html_content = response.text
            if not html_content:
                return "Error: Empty response from ScraperAPI"

            # The transcript data is often embedded in a JSON object in the HTML.
            # We'll use a regex to find this specific JSON data.
            # This is more robust than navigating a constantly changing DOM.
            transcript_pattern = r'"transcriptRenderer".*?"runs":\s*(\[.*?\])'
            match = re.search(transcript_pattern, html_content, re.DOTALL)
            
            if match:
                try:
                    # The content inside the parenthesis is a JSON array
                    runs_data = json.loads(match.group(1))
                    transcript_text = ""
                    
                    # Extract the text from each run
                    for run in runs_data:
                        if 'text' in run:
                            transcript_text += run['text'] + " "
                    
                    if transcript_text.strip():
                        print("Transcript successfully scraped.")
                        return transcript_text.strip()
                    else:
                        return "Error: No transcript text found in scraped data."
                        
                except json.JSONDecodeError as json_error:
                    return f"Error: Failed to parse transcript data JSON: {str(json_error)}"
            else:
                return "Error: Could not find transcript data in the scraped page."
        else:
            return f"Error: ScraperAPI request failed with status code: {response.status_code}"
            
    except Exception as e:
        return f"Error: Failed to fetch transcript with ScraperAPI: {str(e)}"

def fetch_transcript_alternative_method(html_content, video_id):
    """Alternative method to extract transcript from HTML content"""
    if not html_content:
        return "Error: No HTML content provided for alternative extraction"
        
    try:
        # Look for captions/subtitle data in various formats
        patterns = [
            r'"captions".*?"playerCaptionsTracklistRenderer".*?"captionTracks":\s*(\[.*?\])',
            r'"subtitlesTrack".*?"baseUrl":"([^"]+)"',
            r'"captionTracks":\s*(\[.*?\])'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.DOTALL)
            if match:
                try:
                    # Try to extract and process the caption data
                    caption_data = match.group(1) if match.lastindex >= 1 else match.group(0)
                    
                    if 'baseUrl' in caption_data:
                        # Extract the subtitle URL and fetch it
                        url_match = re.search(r'"baseUrl":"([^"]+)"', caption_data)
                        if url_match:
                            subtitle_url = url_match.group(1).replace('\\u0026', '&')
                            return fetch_subtitle_from_url(subtitle_url)
                            
                except Exception as parse_error:
                    print(f"Pattern parsing error: {parse_error}")
                    continue
        
        return "Error: No transcript found in video page. Video may not have captions enabled."
        
    except Exception as e:
        return f"Error: Alternative transcript extraction failed - {str(e)}"

def fetch_subtitle_from_url(subtitle_url):
    """Fetch and parse subtitle content from URL"""
    if not subtitle_url:
        return "Error: No subtitle URL provided"
        
    try:
        if not SCRAPERAPI_KEY:
            return "Error: ScraperAPI key required for subtitle fetching"
            
        params = {
            'api_key': SCRAPERAPI_KEY,
            'url': subtitle_url
        }
        
        session = create_session_with_retry()
        response = session.get("http://api.scraperapi.com", params=params, timeout=30)
        
        if response.status_code == 200:
            # Parse XML subtitle content
            subtitle_content = response.text
            
            if not subtitle_content:
                return "Error: Empty subtitle content received"
            
            # Extract text from XML format
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(subtitle_content)
                transcript_text = ""
                
                for text_element in root.findall('.//text'):
                    if text_element.text:
                        transcript_text += text_element.text + " "
                
                if transcript_text.strip():
                    return transcript_text.strip()
                else:
                    return "Error: No text found in subtitle file"
                
            except ET.ParseError as xml_error:
                # Try regex extraction if XML parsing fails
                try:
                    text_pattern = r'<text[^>]*>(.*?)</text>'
                    matches = re.findall(text_pattern, subtitle_content, re.DOTALL | re.IGNORECASE)
                    
                    if matches:
                        transcript_text = " ".join([re.sub(r'<[^>]+>', '', match) for match in matches])
                        return transcript_text.strip() if transcript_text.strip() else "Error: Empty transcript after regex extraction"
                    else:
                        return f"Error: Failed to extract text from subtitle content. XML parse error: {str(xml_error)}"
                except Exception as regex_error:
                    return f"Error: Both XML parsing and regex extraction failed. XML error: {str(xml_error)}, Regex error: {str(regex_error)}"
        else:
            return f"Error: Failed to fetch subtitle content. Status code: {response.status_code}"
            
    except Exception as e:
        return f"Error: Exception in fetching subtitle from URL - {str(e)}"

def fetch_transcript(video_id):
    """Main transcript fetching function with fallback methods"""
    if not video_id:
        return "Error: Invalid video ID"
        
    try:
        # Method 1: Try direct youtube-transcript-api first (for local development)
        if not SCRAPERAPI_KEY:
            try:
                ytt_api = YouTubeTranscriptApi()
                transcript = ytt_api.fetch(video_id, languages=['en'])
                formatter = TextFormatter()
                result = formatter.format_transcript(transcript)
                return result if result else "Error: Empty transcript returned"
            except Exception as e:
                print(f"Direct method failed: {e}")
                return f"Error: Direct transcript fetch failed - {str(e)}"
    except Exception as direct_error:
        print(f"Direct method setup failed: {direct_error}")
        
    # Method 2: Use ScraperAPI
    if SCRAPERAPI_KEY:
        try:
            scraperapi_result = fetch_transcript_with_scraperapi(video_id)
            if scraperapi_result and not scraperapi_result.startswith("Error") and not scraperapi_result.startswith("ScraperAPI"):
                return scraperapi_result
            else:
                print(f"ScraperAPI method failed: {scraperapi_result}")
        except Exception as scraper_error:
            print(f"ScraperAPI method error: {scraper_error}")
    
    # Method 3: Try with different error handling for youtube-transcript-api
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=['en'])
        # transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        if transcript:
            formatter = TextFormatter()
            result = formatter.format_transcript(transcript)
            return result if result else "Error: Empty transcript after formatting"
        else:
            return "Error: No transcript data retrieved"
    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video"
    except NoTranscriptFound:
        return "Error: No transcript found (even auto-generated)"
    except VideoUnavailable:
        return "Error: Video unavailable"
    except Exception as e:
        error_message = str(e)
        if "blocked" in error_message.lower() or "ip" in error_message.lower():
            return f"Error: IP blocked by YouTube. ScraperAPI integration needed - {error_message}"
        return f"Error: {error_message}"

def search_youtube(query, max_results=5):
    """Search YouTube videos and return results"""
    search_url = f"ytsearch{max_results}:{query}"
    ydl_opts = {'extract_flat': True}
    videos = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(search_url, download=False)
            for entry in results['entries']:
                videos.append({
                    "title": entry.get('title', 'N/A'),
                    "url": f"https://www.youtube.com/watch?v={entry['id']}",
                    "duration": entry.get('duration', 'N/A'),
                    "views": entry.get('view_count', 'N/A')
                })
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"

    return videos

def format_search_results(query, num_results):
    """Format search results for display"""
    if not query.strip():
        return "Please enter a search query."

    try:
        num_results = int(num_results)
        if num_results < 1 or num_results > 10:
            num_results = 5
    except:
        num_results = 5

    videos = search_youtube(query, num_results)

    if isinstance(videos, str):
        return videos

    if not videos:
        return "No videos found."

    formatted_results = []
    for i, video in enumerate(videos, 1):
        duration = video['duration'] if video['duration'] != 'N/A' else 'Unknown'
        views = video['views'] if video['views'] != 'N/A' else 'Unknown'

        result = f"""
        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4><a href="{video['url']}" target="_blank">{video['title']}</a></h4>
            <p><strong>URL:</strong> {video['url']}</p>
            <p><strong>Duration:</strong> {duration} seconds | <strong>Views:</strong> {views}</p>
        </div>
        """
        formatted_results.append(result)

    return "".join(formatted_results)

def extract_video_id(url):
    if "youtube" in url:
        return parse_qs(urlparse(url).query).get("v", [None])[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None

def get_metadata_only(video_url):
    try:
        cmd = ['yt-dlp', '--dump-json', '--no-download', video_url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            metadata = json.loads(result.stdout)
            return {
                'title': metadata.get('title'),
                'description': metadata.get('description'),
                'channel': metadata.get('uploader'),
                'duration': metadata.get('duration'),
                'views': metadata.get('view_count')
            }
        else:
            return {"error": f"yt-dlp failed: {result.stderr.strip()}"}
    except subprocess.TimeoutExpired:
        return {"error": "yt-dlp timed out"}
    except Exception as e:
        return {"error": f"yt-dlp error: {str(e)}"}

def process_youtube_url(url):
    """Fetches transcript, metadata, and prepares the QA chain with memory."""
    global global_qa_chain, global_memory, current_video_title

    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL", {}, "", "Invalid URL - QA chatbot not ready"

    transcript = fetch_transcript(video_id)
    metadata = get_metadata_only(url)

    # Handle None transcript
    if transcript is None:
        transcript = "Error: No transcript could be retrieved"

    # Store video title for context
    current_video_title = metadata.get('title', 'Unknown Video') if isinstance(metadata, dict) else 'Unknown Video'

    # Check if transcript is valid and usable
    transcript_is_valid = (
        isinstance(transcript, str) and 
        transcript.strip() and
        not transcript.startswith("Error") and 
        not transcript.startswith("Transcripts are disabled") and 
        not transcript.startswith("No transcript") and 
        not transcript.startswith("Video unavailable") and 
        not transcript.startswith("IP blocked") and
        not transcript.startswith("ScraperAPI")
    )

    if transcript_is_valid:
        try:
            # Split transcript into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
            chunks = splitter.split_text(transcript)

            if not chunks:
                raise ValueError("No chunks created from transcript")

            # Create vector store
            vector_store = FAISS.from_texts(chunks, embedding_model)

            # Initialize conversation memory with a window of last 10 exchanges
            global_memory = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 conversation turns
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )

            # Create conversational retrieval chain with memory
            global_qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=global_memory,
                return_source_documents=True,
                verbose=True
            )

            status_message = f"✅ QA chatbot is ready! You can now ask questions about '{current_video_title}'"
        except Exception as e:
            global_qa_chain = None
            global_memory = None
            status_message = f"❌ QA setup failed: {str(e)}"
    else:
        global_qa_chain = None
        global_memory = None
        current_video_title = ""
        # Safe string slicing with None check
        transcript_preview = str(transcript)[:100] if transcript else "No transcript available"
        status_message = f"❌ QA chatbot not ready. Issue: {transcript_preview}..."

    # Safe transcript preview
    if isinstance(transcript, str) and len(transcript) > 500:
        short_transcript = transcript[:500] + "..."
    else:
        short_transcript = str(transcript) if transcript else "No transcript available"

    return transcript, metadata, short_transcript, status_message

def get_comments(video_id, api_key, max_results=20):
    """Get comments from a YouTube video"""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_results,
            textFormat='plainText'
        )
        response = request.execute()

        comments = []
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        return comments
    except Exception as e:
        return [f"Error fetching comments: {str(e)}"]

def analyze_comments(comments):
    """Analyze sentiment of comments"""
    if not sentiment_pipe:
        return [(comment, {"label": "UNKNOWN", "score": 0}) for comment in comments]

    try:
        results = sentiment_pipe(comments, truncation=True)
        return list(zip(comments, results))
    except Exception as e:
        return [(comment, {"label": "ERROR", "score": 0}) for comment in comments]

def generate_summary_and_anchors(transcript_text):
    """Generate summary and anchor comments using Gemini"""
    try:
        prompt = f"""
        You are an intelligent assistant. Given a video transcript, generate the following:

        1. A short summary (2-3 sentences).
        2. 3 examples of *relevant* viewer comments based on the video.
        3. 3 examples of *irrelevant* viewer comments.
        4. 3 examples of *spammy* viewer comments.

        Return your output in this JSON format:

        {{
          "summary": "...",
          "relevant": ["...", "...", "..."],
          "irrelevant": ["...", "...", "..."],
          "spammy": ["...", "...", "..."]
        }}

        Transcript:
        \"\"\"
        {transcript_text[:2000]}
        \"\"\"
        """

        model_gen = genai.GenerativeModel("models/gemini-2.0-flash-exp")
        response = model_gen.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating anchors: {str(e)}"

def clean_gemini_output(raw_output):
    """Clean the generated anchor comments"""
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip(), flags=re.MULTILINE)
    return cleaned

def match_comments_to_anchors(user_comments, anchor_dict, top_k=1):
    """Match comments to anchor categories"""
    if not model or not anchor_dict:
        return []

    try:
        all_anchors = []
        anchor_meta = []

        for category in ["relevant", "irrelevant", "spammy"]:
            for text in anchor_dict.get(category, []):
                all_anchors.append(text)
                anchor_meta.append(category)

        if not all_anchors:
            return []

        anchor_embeddings = model.encode(all_anchors, convert_to_tensor=True)

        results = []
        for comment in user_comments:
            comment_embedding = model.encode(comment, convert_to_tensor=True)

            cos_scores = util.cos_sim(comment_embedding, anchor_embeddings)[0]
            top_result = cos_scores.argmax().item()
            match = {
                "comment": comment,
                "matched_anchor": all_anchors[top_result],
                "category": anchor_meta[top_result],
                "score": float(cos_scores[top_result])
            }
            results.append(match)

        return results
    except Exception as e:
        return [{"comment": str(e), "matched_anchor": "", "category": "error", "score": 0}]

def process_comments(video_url, max_comments):
    """Process comments for a YouTube video"""
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid YouTube URL", "No comments to analyze", gr.Dropdown(choices=[], visible=False), "", ""

    comments = get_comments(video_id, API_KEY, max_comments)

    if not comments or (len(comments) == 1 and "Error" in comments[0]):
        return str(comments), "No comments to analyze", gr.Dropdown(choices=[], visible=False), "", ""

    global stored_comments, stored_sentiment_results
    stored_comments = comments
    stored_sentiment_results = analyze_comments(comments)

    comment_choices = [f"{i+1}. {comment[:80]}..." if len(comment) > 80 else f"{i+1}. {comment}"
                      for i, comment in enumerate(comments)]

    comments_summary = f"Successfully fetched {len(comments)} comments!\n\nSelect a comment from the dropdown below to see detailed analysis."

    return (
        "\n".join([f"{i+1}. {comment}" for i, comment in enumerate(comments)]),
        comments_summary,
        gr.Dropdown(choices=comment_choices, label="Select a Comment for Analysis", visible=True, value=None),
        "",  # Clear individual sentiment
        ""   # Clear individual category
    )

stored_comments = []
stored_sentiment_results = []
stored_anchor_dict = {}
stored_matched_results = []

def analyze_selected_comment(selected_comment_text):
    """Analyze the selected comment for sentiment and categorization"""
    if not selected_comment_text or not stored_comments:
        return "No comment selected", "No categorization available"

    try:
        comment_index = int(selected_comment_text.split('.')[0]) - 1

        if comment_index < 0 or comment_index >= len(stored_comments):
            return "Invalid comment selection", "Invalid comment selection"

        selected_comment = stored_comments[comment_index]

        if comment_index < len(stored_sentiment_results):
            _, sentiment = stored_sentiment_results[comment_index]
            label = sentiment.get('label', 'UNKNOWN')
            score = sentiment.get('score', 0)
            sentiment_text = f"Sentiment Analysis:\n\n{label} (Confidence: {score:.3f})\n\nComment:\n{selected_comment}"
        else:
            sentiment_text = f"Comment:\n{selected_comment}\n\nSentiment: Not analyzed"

        category_text = "Click 'Generate Anchor Analysis' first to see categorization"
        if stored_matched_results:
            for result in stored_matched_results:
                if result['comment'] == selected_comment:
                    category_text = f"""AI Categorization:

Category: {result['category'].upper()}
Similarity Score: {result['score']:.3f}

Matched Pattern:
{result['matched_anchor']}

Explanation: This comment was categorized as '{result['category']}' based on semantic similarity to AI-generated anchor patterns."""
                    break

        return sentiment_text, category_text

    except Exception as e:
        return f"Error analyzing comment: {str(e)}", "Error in categorization"

def analyze_with_anchors(video_url, transcript):
    """Generate anchor analysis for comments"""
    global stored_anchor_dict, stored_matched_results

    # Handle None or invalid transcript
    if not transcript or transcript is None:
        return "Error: No transcript available for anchor analysis. Please extract transcript first in Step 2.", "No analysis available"
    
    transcript_str = str(transcript)
    if (transcript_str.startswith("Error") or 
        "No transcript" in transcript_str or 
        transcript_str.startswith("Invalid") or
        len(transcript_str.strip()) < 10):
        return f"Error: Need valid transcript for anchor analysis. Current status: {transcript_str[:100]}...", "No analysis available"

    if not stored_comments:
        return "Error: Need comments first. Please fetch comments using 'Get Comments' button.", "No analysis available"

    try:
        anchor_output = generate_summary_and_anchors(transcript_str)
        if not anchor_output or anchor_output.startswith("Error"):
            return f"Error: Failed to generate anchor analysis - {anchor_output}", "Analysis failed"
            
        cleaned_output = clean_gemini_output(anchor_output)

        try:
            stored_anchor_dict = json.loads(cleaned_output)

            stored_matched_results = match_comments_to_anchors(stored_comments, stored_anchor_dict)

            category_counts = {"relevant": 0, "irrelevant": 0, "spammy": 0}
            for result in stored_matched_results:
                category = result.get('category', 'unknown')
                if category in category_counts:
                    category_counts[category] += 1

            summary_text = f"""✅ Analysis Complete!

Categorization Summary:
- 🎯 Relevant: {category_counts['relevant']} comments
- ❌ Irrelevant: {category_counts['irrelevant']} comments
- 🚫 Spammy: {category_counts['spammy']} comments

Generated Anchor Patterns:
{json.dumps(stored_anchor_dict, indent=2)}"""

            instructions = """✅ Analysis ready!

Now select any comment from the dropdown above to see:
- Detailed sentiment analysis
- AI categorization with similarity score
- Explanation of why it was categorized that way"""

            return summary_text, instructions

        except json.JSONDecodeError as e:
            return f"Error: JSON parsing failed - {str(e)}\n\nRaw output:\n{cleaned_output}", "Analysis failed"
            
    except Exception as e:
        return f"Error: Exception in anchor analysis - {str(e)}", "Analysis failed"

def respond(message, chat_history):
    """Handle chat responses with memory"""
    if global_qa_chain is None:
        error_msg = "Please first extract a video transcript in the 'Analyze Video' tab."
        chat_history.append((message, error_msg))
        return chat_history, ""

    if not message.strip():
        return chat_history, ""

    try:
        # Add context about the video to the query
        contextual_message = f"Based on the video '{current_video_title}', {message}"

        # Use the conversational chain which maintains memory
        result = global_qa_chain({"question": contextual_message})

        # Extract the answer from the result
        answer = result.get("answer", "I couldn't find a relevant answer.")

        # Add source information if available
        if "source_documents" in result and result["source_documents"]:
            answer += f"\n\n*Based on {len(result['source_documents'])} relevant sections from the transcript.*"

        chat_history.append((message, answer))
        return chat_history, ""

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        chat_history.append((message, error_msg))
        return chat_history, ""

def clear_chat():
    """Clear chat history and reset memory"""
    global global_memory
    if global_memory:
        global_memory.clear()
    return None

def get_chat_summary():
    """Get a summary of the current chat session"""
    if global_memory is None:
        return "No chat session active."

    try:
        # Get the conversation history from memory
        messages = global_memory.chat_memory.messages
        if not messages:
            return "No conversation history yet."

        # Count the exchanges
        human_messages = [msg for msg in messages if hasattr(msg, 'content') and msg.__class__.__name__ == 'HumanMessage']
        ai_messages = [msg for msg in messages if hasattr(msg, 'content') and msg.__class__.__name__ == 'AIMessage']

        return f"Chat Session: {len(human_messages)} questions asked about '{current_video_title}'"

    except Exception as e:
        return f"Error getting chat summary: {str(e)}"

# Create the Gradio interface
with gr.Blocks(theme="soft", title="🎬 YouTube Video Analyzer") as demo:
    gr.Markdown("""
    # YouTube Video Analyzer
    Easily analyze YouTube videos for transcripts, metadata, sentiment, viewer comment quality and chat with data!
    
    **Now with ScraperAPI integration to bypass IP blocking!**

    ---
    """)

    with gr.Tabs():
        with gr.Tab("Search Videos"):
            gr.Markdown("## Search YouTube Videos")

            with gr.Row():
                with gr.Column(scale=3):
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your YouTube search query...",
                        lines=1
                    )
                with gr.Column(scale=1):
                    num_results = gr.Slider(
                        label="Number of Results",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1
                    )

            search_button = gr.Button("Search YouTube", variant="primary")

            with gr.Accordion("View Search Results", open=True):
                results_output = gr.HTML(
                    label="Search Results",
                    value="Enter a search query and click 'Search YouTube' to see results."
                )

            search_button.click(
                fn=format_search_results,
                inputs=[search_input, num_results],
                outputs=results_output
            )

            search_input.submit(
                fn=format_search_results,
                inputs=[search_input, num_results],
                outputs=results_output
            )

        with gr.Tab("Analyze Video"):
            gr.Markdown("## Analyze a YouTube Video")

            with gr.Row():
                video_url_input = gr.Textbox(
                    label="YouTube Video URL",
                    placeholder="Paste YouTube video URL here..."
                )
                analyze_button = gr.Button("Extract Metadata & Transcript", variant="primary")

            with gr.Accordion("Transcript and Metadata", open=True):
                with gr.Row():
                    transcript_output = gr.Textbox(label="Full Transcript", lines=8, interactive=False)
                    metadata_output = gr.JSON(label="Video Metadata")

                short_transcript_display = gr.Textbox(label="Preview of Transcript", lines=4, interactive=False)

            # Define chat_status_display before using it
            chat_status_display = gr.Textbox(
                label="QA Chatbot Status",
                lines=1,
                interactive=False,
                value="Extract a transcript to enable the chatbot."
            )

            analyze_button.click(
                fn=process_youtube_url,
                inputs=video_url_input,
                outputs=[transcript_output, metadata_output, short_transcript_display, chat_status_display]
            )

        with gr.Tab("Comment Analysis"):
            gr.Markdown("## Comment Analysis")

            with gr.Row():
                with gr.Column():
                    comment_url_input = gr.Textbox(
                        label="YouTube Video URL for Comments",
                        placeholder="Paste the same or different YouTube URL..."
                    )
                    max_comments = gr.Slider(
                        label="Max Comments to Fetch",
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5
                    )
                    get_comments_btn = gr.Button("Get Comments & Analyze", variant="primary")

            with gr.Accordion("Fetched Comments", open=False):
                with gr.Row():
                    comments_display = gr.Textbox(
                        label="All Comments (Raw)",
                        lines=8,
                        interactive=False
                    )
                    comments_summary = gr.Textbox(
                        label="Comments Status",
                        lines=8,
                        interactive=False,
                        value="Click 'Get Comments & Analyze' to fetch comments."
                    )

            gr.Markdown("### Individual Comment Analysis")

            selected_comment = gr.Dropdown(
                label="Select a Comment for Detailed Analysis",
                choices=[],
                visible=False,
                interactive=True
            )

            with gr.Row():
                individual_sentiment = gr.Textbox(
                    label="Sentiment Analysis",
                    lines=6,
                    interactive=False
                )
                individual_category = gr.Textbox(
                    label="AI Categorization",
                    lines=6,
                    interactive=False
                )

            gr.Markdown("### AI-Powered Comment Categorization")

            analyze_anchors_btn = gr.Button("Generate Anchor Analysis", variant="secondary")

            with gr.Accordion("AI Analysis Results", open=False):
                with gr.Row():
                    anchor_summary = gr.Textbox(
                        label="AI Summary",
                        lines=8,
                        interactive=False
                    )
                    analysis_instructions = gr.Textbox(
                        label="Instructions",
                        lines=8,
                        interactive=False,
                        value="1. Fetch comments first\n2. Generate AI anchor analysis\n3. Select comment to view categorization"
                    )

            get_comments_btn.click(
                fn=process_comments,
                inputs=[comment_url_input, max_comments],
                outputs=[comments_display, comments_summary, selected_comment, individual_sentiment, individual_category]
            )

            analyze_anchors_btn.click(
                fn=analyze_with_anchors,
                inputs=[comment_url_input, transcript_output],
                outputs=[anchor_summary, analysis_instructions]
            )

            selected_comment.change(
                fn=analyze_selected_comment,
                inputs=[selected_comment],
                outputs=[individual_sentiment, individual_category]
            )

        with gr.Tab("Chat with Transcript"):
            gr.Markdown("## Ask questions about the video transcript")
            gr.Markdown("This chat has **memory** - it remembers your previous questions and can build on the conversation!")

            # Chat status and summary
            with gr.Row():
                with gr.Column(scale=2):
                    # chat_status_display is already defined above
                    pass
                with gr.Column(scale=1):
                    chat_summary_btn = gr.Button("View Chat Summary", variant="secondary")
                    chat_summary_output = gr.Textbox(
                        label="Chat Summary",
                        lines=2,
                        interactive=False,
                        value="No active chat session"
                    )

            # Main chat interface
            chatbot = gr.Chatbot(
                label="Chat History",
                height=400,
                show_label=True,
                bubble_full_width=False
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask me about the video content...",
                    scale=4,
                    lines=1
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("Clear Chat & Memory", variant="secondary")

            # Event handlers for chat
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
            send_btn.click(respond, [msg, chatbot], [chatbot, msg])
            clear_btn.click(clear_chat, None, chatbot, queue=False)

            # Chat summary handler
            chat_summary_btn.click(get_chat_summary, None, chat_summary_output)

            gr.Markdown("""
            ### Chat Features:
            - **Conversational Memory**: The AI remembers your previous questions and can reference them
            - **Context Awareness**: Questions are automatically contextualized with the video title
            - **Source References**: Answers include information about which parts of the transcript were used
            - **Memory Window**: Keeps track of your last 10 conversation exchanges
            """)

# Add this at the very end of your app.py file, replacing the existing launch line:

if __name__ == "__main__":
    import os
    
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.environ.get("PORT", 8080))
    
    # For local development
    if os.environ.get("GAE_ENV", "").startswith("standard") or os.environ.get("GOOGLE_CLOUD_PROJECT"):
        # Running on Google Cloud
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            debug=False,
            show_error=True
        )
    else:
        # Running locally
        demo.launch(
            debug=True,
            inbrowser=True
        )