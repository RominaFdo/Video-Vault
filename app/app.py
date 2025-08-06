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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables - handle both local and Cloud Run environments
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.info("dotenv not available, using environment variables directly")

# Get API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables")
if not YOUTUBE_API_KEY:
    logger.error("YOUTUBE_API_KEY not found in environment variables")

# Configure APIs
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)

API_KEY = YOUTUBE_API_KEY

# Initialize models with error handling
global_qa_chain = None
global_memory = None
current_video_title = ""
sentiment_pipe = None
model = None
llm = None
embedding_model = None

def initialize_models():
    """Initialize ML models with error handling"""
    global sentiment_pipe, model, llm, embedding_model
    
    try:
        logger.info("Initializing models...")
        
        if GOOGLE_API_KEY:
            llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.0-flash-exp",  # Use faster model for Cloud Run
                temperature=0.3, 
                google_api_key=GOOGLE_API_KEY
            )
            logger.info("LLM initialized successfully")
        
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("Embedding model initialized")
        
        sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", 
            device=-1  # Use CPU
        )
        logger.info("Sentiment pipeline initialized")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer initialized")
        
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        # Continue without models - the app can still function partially

# Initialize models on startup
initialize_models()

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
        logger.error(f"YouTube search error: {e}")
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

def fetch_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=['en'])
        formatter = TextFormatter()
        return formatter.format_transcript(transcript)
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "No transcript found (even auto)."
    except VideoUnavailable:
        return "Video unavailable."
    except Exception as e:
        logger.error(f"Transcript fetch error: {e}")
        return f"Error: {str(e)}"

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
        logger.error(f"Metadata extraction error: {e}")
        return {"error": f"yt-dlp error: {str(e)}"}

def process_youtube_url(url):
    """Fetches transcript, metadata, and prepares the QA chain with memory."""
    global global_qa_chain, global_memory, current_video_title
    
    if not llm or not embedding_model:
        return "Service temporarily unavailable - models not loaded", {}, "", "QA chatbot unavailable - API keys missing or models failed to load"
    
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL", {}, "", None

    transcript = fetch_transcript(video_id)
    metadata = get_metadata_only(url)
    
    # Store video title for context
    current_video_title = metadata.get('title', 'Unknown Video') if isinstance(metadata, dict) else 'Unknown Video'

    if isinstance(transcript, str) and not "Error" in transcript:
        try:
            # Split transcript into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
            chunks = splitter.split_text(transcript)

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
            
            status_message = f"QA chatbot is ready! You can now ask questions about '{current_video_title}'"
        except Exception as e:
            logger.error(f"QA chain creation error: {e}")
            global_qa_chain = None
            global_memory = None
            status_message = f"QA chatbot setup failed: {str(e)}"
    else:
        global_qa_chain = None
        global_memory = None
        current_video_title = ""
        status_message = "QA chatbot is not ready. No transcript found."

    short_transcript = transcript[:500] + "..." if isinstance(transcript, str) and len(transcript) > 500 else transcript

    return transcript, metadata, short_transcript, status_message

def get_comments(video_id, api_key, max_results=20):
    """Get comments from a YouTube video"""
    if not api_key:
        return ["Error: YouTube API key not configured"]
        
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
        logger.error(f"Comment fetch error: {e}")
        return [f"Error fetching comments: {str(e)}"]

def analyze_comments(comments):
    """Analyze sentiment of comments"""
    if not sentiment_pipe:
        return [(comment, {"label": "UNAVAILABLE", "score": 0}) for comment in comments]

    try:
        results = sentiment_pipe(comments, truncation=True)
        return list(zip(comments, results))
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return [(comment, {"label": "ERROR", "score": 0}) for comment in comments]

def generate_summary_and_anchors(transcript_text):
    """Generate summary and anchor comments using Gemini"""
    if not GOOGLE_API_KEY:
        return "Error: Google API key not configured"
        
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
        logger.error(f"Anchor generation error: {e}")
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
        logger.error(f"Comment matching error: {e}")
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
        logger.error(f"Comment analysis error: {e}")
        return f"Error analyzing comment: {str(e)}", "Error in categorization"

def analyze_with_anchors(video_url, transcript):
    """Generate anchor analysis for comments"""
    global stored_anchor_dict, stored_matched_results

    if not transcript or "No transcript" in transcript:
        return "Need transcript for anchor analysis. Please extract transcript first in Step 2.", "No analysis available"

    if not stored_comments:
        return "Need comments first. Please fetch comments using 'Get Comments' button.", "No analysis available"

    anchor_output = generate_summary_and_anchors(transcript)
    cleaned_output = clean_gemini_output(anchor_output)

    try:
        stored_anchor_dict = json.loads(cleaned_output)

        stored_matched_results = match_comments_to_anchors(stored_comments, stored_anchor_dict)

        category_counts = {"relevant": 0, "irrelevant": 0, "spammy": 0}
        for result in stored_matched_results:
            category = result.get('category', 'unknown')
            if category in category_counts:
                category_counts[category] += 1

        summary_text = f"""Analysis Complete!

Categorization Summary:
- 🎯 Relevant: {category_counts['relevant']} comments
- ❌ Irrelevant: {category_counts['irrelevant']} comments
- 🚫 Spammy: {category_counts['spammy']} comments

Generated Anchor Patterns:
{json.dumps(stored_anchor_dict, indent=2)}"""

        instructions = """Analysis ready!

Now select any comment from the dropdown above to see:
- Detailed sentiment analysis
- AI categorization with similarity score
- Explanation of why it was categorized that way"""

        return summary_text, instructions

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return f"JSON Error: {e}\n\nRaw output:\n{cleaned_output}", "Analysis failed"

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
        logger.error(f"Chat response error: {e}")
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
        logger.error(f"Chat summary error: {e}")
        return f"Error getting chat summary: {str(e)}"

# Create the Gradio interface
def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(theme="soft", title="🎬 YouTube Video Analyzer") as demo:
        gr.Markdown("""
        # 🎬 YouTube Video Analyzer
        Easily analyze YouTube videos for transcripts, metadata, sentiment, viewer comment quality and chat with data!

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

    return demo

# Initialize the interface
demo = create_interface()

if __name__ == "__main__":
    # Get port from environment (Cloud Run sets this)
    port = int(os.environ.get("PORT", 7860))
    
    logger.info(f"Starting Gradio app on port {port}")
    
    # Launch with Cloud Run compatible settings
    demo.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=port,       # Use Cloud Run's assigned port
        share=False,           # Don't create public Gradio link
        # show_error=True,       # Show errors for debugging
        enable_queue=True,     # Enable queue for better performance
        # favicon_path=None,     # Disable favicon to reduce requests
        # show_api=False         # Disable API docs to reduce memory
    )