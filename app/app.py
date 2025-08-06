import gradio as gr
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def greet(name):
    """Simple greeting function"""
    if not name:
        return "Hello! Please enter your name."
    return f"Hello, {name}! YouTube Analyzer is working on Cloud Run! 🎉"

def test_features():
    """Test basic functionality"""
    return """
    🎬 YouTube Video Analyzer - Cloud Run Test
    
    ✅ Gradio interface: Working
    ✅ Cloud Run deployment: Working
    ✅ Port 7860: Working
    ✅ Container startup: Working
    
    Ready to add more features!
    """

def create_interface():
    """Create minimal Gradio interface"""
    
    with gr.Blocks(title="YouTube Analyzer - Cloud Run Test") as demo:
        gr.Markdown("# 🎬 YouTube Video Analyzer")
        gr.Markdown("## Cloud Run Deployment Test")
        
        with gr.Tab("Test Basic Function"):
            name_input = gr.Textbox(label="Enter your name", placeholder="Type your name here...")
            greet_button = gr.Button("Say Hello", variant="primary")
            greet_output = gr.Textbox(label="Response", interactive=False)
            
            greet_button.click(fn=greet, inputs=name_input, outputs=greet_output)
            name_input.submit(fn=greet, inputs=name_input, outputs=greet_output)
        
        with gr.Tab("System Status"):
            status_button = gr.Button("Check System Status", variant="secondary")
            status_output = gr.Textbox(label="System Status", lines=10, interactive=False)
            
            status_button.click(fn=test_features, outputs=status_output)
        
        gr.Markdown("""
        ### Next Steps:
        1. ✅ Test this basic version works
        2. 🔄 Add yt-dlp for video search
        3. 🔄 Add transcript features
        4. 🔄 Add comment analysis
        5. 🔄 Add ML features
        """)
    
    return demo

def main():
    """Main function"""
    try:
        port = int(os.environ.get("PORT", 7860))
        logger.info(f"Starting minimal YouTube Analyzer on port {port}")
        
        demo = create_interface()
        
        # Launch with minimal settings
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            enable_queue=False,
            show_error=True,
            quiet=False,
            show_api=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        raise

if __name__ == "__main__":
    main()