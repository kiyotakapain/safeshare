# SafeShare - Toxic-Free Social Platform

A social platform with AI-powered toxicity detection using Claude API.

## Features
- User authentication
- Post and comment creation
- AI-based toxicity detection
- User notifications
- Automatic content moderation

## Tech Stack
- Flask (Backend)
- SQLite/PostgreSQL (Database)
- Claude AI API (Toxicity detection)

## Local Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.env` file with your API keys
6. Run: `python app.py`

## Environment Variables
- `SECRET_KEY`: Flask secret key
- `ANTHROPIC_API_KEY`: Your Claude API key

## Deployment
Deployed on [Render/Railway/Heroku]
