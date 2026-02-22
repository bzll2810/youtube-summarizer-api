# YouTube Summarizer API

A FastAPI-based service that generates AI summaries of YouTube videos using LaMini-Flan-T5.

## 🚀 Features
- Extract transcripts from any YouTube video
- Generate abstractive summaries using AI
- REST API for Chrome extension integration

## 📡 API Endpoints
- `GET /health` - Check if API is running
- `POST /summarize` - Generate summary (requires JSON with `videoId`)

## 🛠️ Deployment
Deploy on Render using this repository.

## 📄 License
MIT