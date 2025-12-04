# Website Backend

FastAPI-based backend server for the ISRO-GeoNLI web interface providing user authentication, chat history management, and integration with the main ISRO-GeoNLI API.

## Overview

This backend service handles:
- User authentication and JWT token management
- Chat session and message history storage
- Image upload to AWS S3
- Integration with external APIs (Gemini AI, ISRO-GeoNLI pipeline)
- CORS middleware for frontend communication

## Architecture

```
website-backend/
├── main.py              # FastAPI application entry point
├── db.py                # MongoDB connection setup
├── routers/
│   ├── login.py         # Authentication endpoints
│   ├── chat.py          # Chat and query processing
│   ├── chatHistory.py   # Session history management
│   └── getHistory.py    # Message retrieval
```

## Prerequisites

- Python 3.10+
- MongoDB instance (local or cloud)
- AWS S3 bucket (for image storage)

## Installation

### Step 1: Navigate to Backend Directory

```bash
cd website-backend
```

### Step 2: Install Dependencies

```bash
pip install fastapi uvicorn motor pymongo python-jose python-multipart boto3 python-dotenv httpx pillow
```

### Step 3: Configure Environment Variables

Create a `.env` file in the `website-backend` directory:

```env
# MongoDB Configuration
DATABASE_URL=mongodb://localhost:27017

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
AWS_BUCKET_NAME=your_bucket_name

# JWT Configuration
JWT_SECRET=your_jwt_secret_key

# Google Gemini API (optional)
GEMINI_API_KEY=your_gemini_api_key

# ISRO-GeoNLI API
ISRO_API_URL=http://localhost:8080
```

## Running the Server

### Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

The server will be available at `http://localhost:8001`

## API Endpoints

### Authentication

**POST /api/login**
- Login user and receive JWT token
- Request: `{"email": "user@example.com", "password": "password"}`
- Response: `{"access_token": "jwt_token", "token_type": "bearer"}`

**POST /api/signup**
- Register new user
- Request: `{"email": "user@example.com", "password": "password", "name": "User Name"}`
- Response: `{"message": "User created successfully"}`

### Chat

**POST /api/chat**
- Send query with image to ISRO-GeoNLI pipeline
- Request: Form data with `instruction` (text) and `file` (image)
- Response: Query results with annotated image
- Requires: JWT Bearer token

### History

**GET /api/sessions**
- Get all chat sessions for authenticated user
- Requires: JWT Bearer token
- Response: List of session objects

**GET /api/messages/{session_id}**
- Get all messages in a specific session
- Requires: JWT Bearer token
- Response: List of message objects

**POST /api/session**
- Create new chat session
- Request: `{"session_name": "My Session"}`
- Requires: JWT Bearer token
- Response: `{"session_id": "..."}`

## Database Schema

### Collections

**users**
```json
{
  "_id": "ObjectId",
  "email": "user@example.com",
  "password": "hashed_password",
  "name": "User Name",
  "created_at": "timestamp"
}
```

**sessions**
```json
{
  "_id": "ObjectId",
  "user_email": "user@example.com",
  "session_name": "My Session",
  "created_at": "timestamp"
}
```

**messages**
```json
{
  "_id": "ObjectId",
  "session_id": "session_id",
  "user_email": "user@example.com",
  "query": "User query text",
  "image_url": "s3://bucket/image.jpg",
  "response": "API response",
  "timestamp": "timestamp"
}
```

## Configuration

### CORS Settings

The backend allows requests from:
```python
origins = [
    "https://isrogeonli.in:5173",
]
```

Update `main.py` to add your frontend URL:
```python
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]
```

### JWT Token Expiry

Default: 24 hours. Modify in `routers/login.py`:
```python
expire = datetime.utcnow() + timedelta(hours=24)
```

## Integration with Main API

The backend communicates with the ISRO-GeoNLI API (port 8080):

```python
# In routers/chat.py
ISRO_API_URL = "http://localhost:8080/query"
response = requests.post(ISRO_API_URL, json=payload)
```

Ensure the main ISRO-GeoNLI API server is running before starting this backend.

## Security Notes

1. **JWT Secret**: Use a strong, randomly generated secret key
2. **Password Hashing**: Passwords are hashed using bcrypt
3. **CORS**: Restrict `allow_origins` to your actual frontend domains
4. **AWS Credentials**: Never commit credentials to version control
5. **MongoDB**: Use authentication in production environments

## Troubleshooting

### MongoDB Connection Failed

```bash
# Check MongoDB is running
# Linux/Mac
sudo systemctl status mongod

# Windows
net start MongoDB

# Or use MongoDB Atlas (cloud)
DATABASE_URL=mongodb+srv://username:password@cluster.mongodb.net/dbname
```

### AWS S3 Upload Failed

- Verify AWS credentials in `.env`
- Check bucket permissions (PutObject, GetObject)
- Ensure bucket name is correct

### CORS Errors

Add your frontend URL to `origins` in `main.py`:
```python
origins = [
    "http://localhost:5173",  # Vite dev server
]
```

### JWT Token Invalid

- Ensure `JWT_SECRET` matches between login and verification
- Check token hasn't expired
- Verify token format in Authorization header: `Bearer <token>`

## Development

### Adding New Endpoints

1. Create router in `routers/` directory
2. Import and include in `main.py`:
```python
from routers.new_router import router as new_router
app.include_router(new_router, prefix="/api", tags=["New"])
```

### Database Operations

Use Motor (async MongoDB driver):
```python
from db import db

collection = db["collection_name"]
result = await collection.find_one({"key": "value"})
```

## License

Part of the ISRO-GeoNLI project.
