from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re
import uvicorn

app = FastAPI()

# Enable CORS for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a MUCH SMALLER model (fits in 512MB RAM)
print("Loading lightweight model...")
try:
    # Using distilbart-cnn-12-6 - much smaller than LaMini
    summarizer = pipeline("summarization", 
                         model="sshleifer/distilbart-cnn-12-6")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    summarizer = None

class VideoRequest(BaseModel):
    videoId: str

def extract_video_id(url):
    """Extract video ID from any YouTube URL"""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[?&]|$)",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"^([0-9A-Za-z_-]{11})$"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    """Fetch transcript from YouTube"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([item['text'] for item in transcript])
        text = re.sub(r'\s+', ' ', text).strip()
        return {"success": True, "text": text}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/")
def root():
    return {"message": "YouTube Summarizer API is running", "status": "active"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": summarizer is not None}

@app.post("/summarize")
def summarize(request: VideoRequest):
    # Get transcript
    transcript_result = get_transcript(request.videoId)
    
    if not transcript_result["success"]:
        return {"success": False, "error": transcript_result["error"]}
    
    text = transcript_result["text"]
    
    # Truncate if too long
    if len(text) > 1024:
        text = text[:1024]
    
    # Generate summary
    try:
        summary = summarizer(text, 
                           max_length=150, 
                           min_length=40,
                           do_sample=False)[0]['summary_text']
        
        return {
            "success": True,
            "summary": summary,
            "videoId": request.videoId
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
