---
title: VideoVault
emoji: 🏢
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.41.0
app_file: app.py
pinned: false
license: mit
short_description: '"Search, Analyze, Chat, Understand" - Suggests a treasure tr'
---
# 🎬 YouTube Video Analyzer (Gradio + Gemini + LangChain)

This app lets you:
- ✅ Analyze YouTube video transcripts
- ✅ Chat with the video using Gemini
- ✅ Fetch & classify comments (Relevant / Irrelevant / Spammy)
- ✅ Perform sentiment analysis

Built using Gradio, Gemini Pro 2.5, LangChain, and FAISS.

## Run Locally

```bash
pip install -r requirements.txt
cd app
python app.py

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
