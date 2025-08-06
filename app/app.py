import gradio as gr
import yt_dlp
from urllib.parse import urlparse, parse_qs
import json
import subprocess
import os
import logging
import traceback
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_import(module_name, package_name=None):
    """Safely import modules and log if they fail"""
    try:
        if package_name:
            module = __import__(module_name, fromlist=[package_name])
            return getattr(module, package_name)
        else:
            return __import__(module_name)
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        return None

def error_handler(func):
    """Decorator to handle errors gracefully"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            return f"Error: {str(e)}"
    return wrapper

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.info("dotenv not available, using environment variables directly")

# Get API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables")
if not YOUTUBE_API_KEY:
    logger.warning("YOUTUBE_API_KEY not found in environment variables")

# Initialize optional imports
youtube_transcript_api = safe_import("youtube_transcript_api")
googleapiclient = safe_import("googleapiclient.discovery")
transformers = safe_import("transformers")
google_genai = safe_import("google.generativeai")
sentence_transformers = safe_import("sentence_transformers")
langchain_modules = {}

# Try importing langchain modules
langchain_imports = [
    ("langchain.text_splitter", "RecursiveCharacterTextSplitter"),
    ("langchain.chains", "ConversationalRetrievalChain"),
    ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    ("langchain.vectorstores", "FAISS"),
    ("langchain.embeddings", "SentenceTransformerEmbeddings"),
    ("langchain.memory", "ConversationBufferWindowMemory")
]

for module_name, class_name in langchain_imports:
    result = safe_import(module_name, class_name)
    if result:
        langchain_modules[class_name] = result

# Configure APIs if available
if GOOGLE_API_KEY and google_genai:
    try:
        google_genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Google AI configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Google AI: {e}")

# Global variables
global_qa_chain = None
global_memory = None
current_video_title = ""
sentiment_pipe = None
model = None
llm = None
embedding_model = None
stored_comments = []
stored_sentiment_results = []
stored_anchor_dict = {}
stored_matched_results = []

class ModelManager:
    """Manage model initialization with fallbacks"""
    
    def __init__(self):
        self.models_initialized = False
        self.available_features = {
            'search': True,  # Basic search always available
            'transcript': bool(youtube_transcript_api),
            'metadata': True,  # yt-dlp based
            'comments': bool(googleapiclient and YOUTUBE_API_KEY),
            'sentiment': bool(transformers),
            'qa_chat': bool(langchain_modules and GOOGLE_API_KEY),
            'categorization': bool(sentence_transformers and google_genai)
        }
        
    def initialize_models(self):
        """Initialize models with error handling"""
        global sentiment_pipe, model, llm, embedding_model
        
        logger.info("Initializing models...")
        logger.info(f"Available features: {self.available_features}")
        
        try:
            # Initialize LLM if possible
            if self.available_features['qa_chat']:
                try:
                    llm = langchain_modules['ChatGoogleGenerativeAI'](
                        model="gemini-1.5-flash",  # Use more reliable model
                        temperature=0.3,
                        google_api_key=GOOGLE_API_KEY
                    )
                    logger.info("LLM initialized successfully")
                except Exception as e:
                    logger.error(f"LLM initialization failed: {e}")
                    self.available_features['qa_chat'] = False
            
            # Initialize embedding model if possible
            if self.available_features['qa_chat']:
                try:
                    embedding_model = langchain_modules['SentenceTransformerEmbeddings'](
                        model_name="all-MiniLM-L6-v2"
                    )
                    logger.info("Embedding model initialized")
                except Exception as e:
                    logger.error(f"Embedding model initialization failed: {e}")
                    self.available_features['qa_chat'] = False
            
            # Initialize sentiment pipeline if possible
            if self.available_features['sentiment']:
                try:
                    sentiment_pipe = transformers.pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        device=-1  # Use CPU
                    )
                    logger.info("Sentiment pipeline initialized")
                except Exception as e:
                    logger.error(f"Sentiment pipeline initialization failed: {e}")
                    self.available_features['sentiment'] = False
            
            # Initialize sentence transformer if possible
            if self.available_features['categorization']:
                try:
                    model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Sentence transformer initialized")
                except Exception as e:
                    logger.error(f"Sentence transformer initialization failed: {e}")
                    self.available_features['categorization'] = False
                    
            self.models_initialized = True
            logger.info("Model initialization completed")
            
        except Exception as e:
            logger.error(f"Model initialization error: {e}")

# Initialize model manager
model_manager = ModelManager()

@error_handler
def search_youtube(query, max_results=5):
    """Search YouTube videos and return results"""
    if not model_manager.available_features['search']:
        return "Search feature not available"
        
    search_url = f"ytsearch{max_results}:{query}"
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True
    }
    videos = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(search_url, download=False)
            for entry in results.get('entries', []):
                videos.append({
                    "title": entry.get('title', 'N/A'),
                    "url": f"https://www.youtube.com/watch?v={entry['id']}",
                    "duration": entry.get('duration', 'N/A'),
                    "views": entry.get('view_count', 'N/A')
                })
    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return f"Error searching YouTube: {str(e)}"

    return videos

@error_handler
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
    """Extract video ID from YouTube URL"""
    if "youtube" in url:
        return parse_qs(urlparse(url).query).get("v", [None])[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None

@error_handler
def fetch_transcript(video_id):
    """Fetch transcript using youtube-transcript-api"""
    if not model_manager.available_features['transcript']:
        return "Transcript feature not available - youtube-transcript-api not installed"
        
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
        from youtube_transcript_api.formatters import TextFormatter
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        formatter = TextFormatter()
        return formatter.format_transcript(transcript)
    except Exception as e:
        logger.error(f"Transcript fetch error: {e}")
        return f"Error fetching transcript: {str(e)}"

@error_handler
def get_metadata_only(video_url):
    """Get video metadata using yt-dlp"""
    try:
        cmd = ['yt-dlp', '--dump-json', '--no-download', video_url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            metadata = json.loads(result.stdout)
            return {
                'title': metadata.get('title'),
                'description': metadata.get('description', '')[:500] + "..." if len(metadata.get('description', '')) > 500 else metadata.get('description'),
                'channel': metadata.get('uploader'),
                'duration': metadata.get('duration'),
                'views': metadata.get('view_count')
            }
        else:
            return {"error": f"yt-dlp failed: {result.stderr.strip()}"}
    except subprocess.TimeoutExpired:
        return {"error": "yt-dlp timed out"}
    except Exception as e:
        logger.error(f"Metadata extraction error: {e}")
        return {"error": f"yt-dlp error: {str(e)}"}

@error_handler
def process_youtube_url(url):
    """Process YouTube URL and set up QA chain"""
    global global_qa_chain, global_memory, current_video_title
    
    if not url:
        return "Please enter a YouTube URL", {}, "", "Please enter a YouTube URL"
    
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL", {}, "", "Invalid YouTube URL"

    # Get metadata first (always available)
    metadata = get_metadata_only(url)
    current_video_title = metadata.get('title', 'Unknown Video') if isinstance(metadata, dict) else 'Unknown Video'
    
    # Get transcript if available
    transcript = "Transcript not available"
    if model_manager.available_features['transcript']:
        transcript = fetch_transcript(video_id)
    
    # Set up QA chain if possible
    qa_status = "QA chat not available"
    if model_manager.available_features['qa_chat'] and isinstance(transcript, str) and "Error" not in transcript:
        try:
            # Split transcript into chunks
            splitter = langchain_modules['RecursiveCharacterTextSplitter'](
                chunk_size=512, 
                chunk_overlap=64
            )
            chunks = splitter.split_text(transcript)

            # Create vector store
            vector_store = langchain_modules['FAISS'].from_texts(chunks, embedding_model)

            # Initialize conversation memory
            global_memory = langchain_modules['ConversationBufferWindowMemory'](
                k=10,
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )

            # Create conversational retrieval chain
            global_qa_chain = langchain_modules['ConversationalRetrievalChain'].from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=global_memory,
                return_source_documents=True
            )
            
            qa_status = f"QA chatbot ready for '{current_video_title}'"
            
        except Exception as e:
            logger.error(f"QA chain setup error: {e}")
            global_qa_chain = None
            global_memory = None
            qa_status = f"QA setup failed: {str(e)}"
    
    # Return short transcript for preview
    short_transcript = transcript[:500] + "..." if isinstance(transcript, str) and len(transcript) > 500 else transcript
    
    return transcript, metadata, short_transcript, qa_status

@error_handler
def get_comments(video_id, api_key, max_results=20):
    """Get comments from YouTube video"""
    if not model_manager.available_features['comments']:
        return ["Comments feature not available - YouTube API not configured"]
        
    try:
        from googleapiclient.discovery import build
        
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
        logger.error(f"Comment fetch error: {e}")
        return [f"Error fetching comments: {str(e)}"]

@error_handler
def analyze_comments_sentiment(comments):
    """Analyze sentiment of comments"""
    if not model_manager.available_features['sentiment']:
        return [(comment, {"label": "UNAVAILABLE", "score": 0}) for comment in comments]

    try:
        results = sentiment_pipe(comments, truncation=True)
        return list(zip(comments, results))
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return [(comment, {"label": "ERROR", "score": 0}) for comment in comments]

@error_handler
def process_comments(video_url, max_comments):
    """Process comments for a YouTube video"""
    global stored_comments, stored_sentiment_results
    
    if not video_url:
        return "Please enter a YouTube URL", "No comments", gr.Dropdown(choices=[], visible=False), "", ""
    
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid YouTube URL", "No comments", gr.Dropdown(choices=[], visible=False), "", ""

    comments = get_comments(video_id, YOUTUBE_API_KEY, max_comments)

    if not comments or (len(comments) == 1 and "Error" in comments[0]):
        return str(comments), "No comments available", gr.Dropdown(choices=[], visible=False), "", ""

    stored_comments = comments
    stored_sentiment_results = analyze_comments_sentiment(comments)

    comment_choices = [f"{i+1}. {comment[:80]}..." if len(comment) > 80 else f"{i+1}. {comment}"
                      for i, comment in enumerate(comments)]

    comments_summary = f"Fetched {len(comments)} comments. Select one below for analysis."

    return (
        "\n".join([f"{i+1}. {comment}" for i, comment in enumerate(comments)]),
        comments_summary,
        gr.Dropdown(choices=comment_choices, label="Select Comment", visible=True, value=None),
        "",
        ""
    )

@error_handler
def analyze_selected_comment(selected_comment_text):
    """Analyze selected comment"""
    if not selected_comment_text or not stored_comments:
        return "No comment selected", "No analysis available"

    try:
        comment_index = int(selected_comment_text.split('.')[0]) - 1
        if comment_index < 0 or comment_index >= len(stored_comments):
            return "Invalid selection", "Invalid selection"

        selected_comment = stored_comments[comment_index]
        
        # Sentiment analysis
        sentiment_text = f"Comment: {selected_comment}\n\n"
        if comment_index < len(stored_sentiment_results):
            _, sentiment = stored_sentiment_results[comment_index]
            label = sentiment.get('label', 'UNKNOWN')
            score = sentiment.get('score', 0)
            sentiment_text += f"Sentiment: {label} (Confidence: {score:.3f})"
        else:
            sentiment_text += "Sentiment: Not analyzed"

        return sentiment_text, "Advanced categorization coming soon..."

    except Exception as e:
        logger.error(f"Comment analysis error: {e}")
        return f"Error: {str(e)}", "Analysis failed"

@error_handler
def respond(message, chat_history):
    """Handle chat responses"""
    if not model_manager.available_features['qa_chat']:
        error_msg = "QA chat feature not available - missing dependencies or API keys"
        chat_history.append((message, error_msg))
        return chat_history, ""
        
    if global_qa_chain is None:
        error_msg = "Please extract a video transcript first in the 'Analyze Video' tab"
        chat_history.append((message, error_msg))
        return chat_history, ""

    if not message.strip():
        return chat_history, ""

    try:
        result = global_qa_chain({"question": message})
        answer = result.get("answer", "I couldn't find a relevant answer.")
        
        if "source_documents" in result and result["source_documents"]:
            answer += f"\n\n*Based on {len(result['source_documents'])} sections from transcript*"
        
        chat_history.append((message, answer))
        return chat_history, ""
        
    except Exception as e:
        logger.error(f"Chat response error: {e}")
        error_msg = f"Error: {str(e)}"
        chat_history.append((message, error_msg))
        return chat_history, ""

def clear_chat():
    """Clear chat history"""
    global global_memory
    if global_memory:
        global_memory.clear()
    return None

def get_feature_status():
    """Get status of available features"""
    status = "🎬 YouTube Video Analyzer - Feature Status:\n\n"
    
    features = {
        'Video Search': model_manager.available_features['search'],
        'Metadata Extraction': model_manager.available_features['metadata'],
        'Transcript Extraction': model_manager.available_features['transcript'],
        'Comment Fetching': model_manager.available_features['comments'],
        'Sentiment Analysis': model_manager.available_features['sentiment'],
        'QA Chat': model_manager.available_features['qa_chat'],
        'Comment Categorization': model_manager.available_features['categorization']
    }
    
    for feature, available in features.items():
        status += f"{'✅' if available else '❌'} {feature}\n"
    
    if not model_manager.available_features['comments']:
        status += "\n⚠️ Set YOUTUBE_API_KEY for comment features"
    if not model_manager.available_features['qa_chat']:
        status += "\n⚠️ Set GOOGLE_API_KEY for QA chat features"
        
    return status

def create_interface():
    """Create Gradio interface"""
    
    # Initialize models when creating interface
    if not model_manager.models_initialized:
        model_manager.initialize_models()
    
    with gr.Blocks(theme="soft", title="🎬 YouTube Video Analyzer") as demo:
        
        gr.Markdown("# YouTube Video Analyzer")
        
        # Feature status
        with gr.Accordion("Feature Status", open=False):
            status_display = gr.Textbox(
                value=get_feature_status(),
                lines=10,
                interactive=False,
                label="Available Features"
            )
        
        with gr.Tabs():
            # Search Tab
            with gr.Tab("Search Videos"):
                gr.Markdown("## Search YouTube Videos")
                
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter search terms...",
                        scale=3
                    )
                    num_results = gr.Slider(
                        label="Results",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        scale=1
                    )
                
                search_button = gr.Button("Search YouTube", variant="primary")
                results_output = gr.HTML(label="Results")
                
                search_button.click(
                    fn=format_search_results,
                    inputs=[search_input, num_results],
                    outputs=results_output
                )
            
            # Analyze Tab
            with gr.Tab("Analyze Video"):
                gr.Markdown("## Extract Video Data")
                
                video_url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=..."
                )
                analyze_button = gr.Button("Analyze Video", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        transcript_output = gr.Textbox(
                            label="Transcript",
                            lines=8,
                            interactive=False
                        )
                        chat_status = gr.Textbox(
                            label="QA Status",
                            lines=2,
                            interactive=False
                        )
                    with gr.Column():
                        metadata_output = gr.JSON(label="Metadata")
                        preview_output = gr.Textbox(
                            label="Transcript Preview",
                            lines=4,
                            interactive=False
                        )
                
                analyze_button.click(
                    fn=process_youtube_url,
                    inputs=video_url_input,
                    outputs=[transcript_output, metadata_output, preview_output, chat_status]
                )
            
            # Comments Tab
            with gr.Tab("Comment Analysis"):
                gr.Markdown("## Analyze Video Comments")
                
                with gr.Row():
                    comment_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        scale=3
                    )
                    max_comments = gr.Slider(
                        label="Max Comments",
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5,
                        scale=1
                    )
                
                get_comments_btn = gr.Button("Get Comments", variant="primary")
                
                with gr.Row():
                    comments_display = gr.Textbox(
                        label="Comments",
                        lines=6,
                        interactive=False
                    )
                    comments_status = gr.Textbox(
                        label="Status",
                        lines=6,
                        interactive=False
                    )
                
                selected_comment = gr.Dropdown(
                    label="Select Comment for Analysis",
                    choices=[],
                    visible=False
                )
                
                with gr.Row():
                    sentiment_output = gr.Textbox(
                        label="Sentiment Analysis",
                        lines=4,
                        interactive=False
                    )
                    category_output = gr.Textbox(
                        label="Categorization",
                        lines=4,
                        interactive=False
                    )
                
                get_comments_btn.click(
                    fn=process_comments,
                    inputs=[comment_url, max_comments],
                    outputs=[comments_display, comments_status, selected_comment, sentiment_output, category_output]
                )
                
                selected_comment.change(
                    fn=analyze_selected_comment,
                    inputs=[selected_comment],
                    outputs=[sentiment_output, category_output]
                )
            
            # Chat Tab
            with gr.Tab("Chat with Video"):
                gr.Markdown("## Ask Questions About the Video")
                
                chatbot = gr.Chatbot(height=400, label="Conversation")
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Question",
                        placeholder="Ask about the video content...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                
                msg.submit(respond, [msg, chatbot], [chatbot, msg])
                send_btn.click(respond, [msg, chatbot], [chatbot, msg])
                clear_btn.click(clear_chat, None, chatbot)
    
    return demo

def main():
    """Main function to run the app"""
    try:
        # Get port from environment
        port = int(os.environ.get("PORT", 7860))
        
        logger.info(f"Starting app on port {port}")
        logger.info(f"Available features: {model_manager.available_features}")
        
        # Create and launch interface
        demo = create_interface()
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            enable_queue=True,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start app: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()