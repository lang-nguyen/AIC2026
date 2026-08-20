# AIC 2026 - Smart Retrieval System

## 1. Requirements
- Python 3.10+
- Node.js 18+
- Docker & Docker Desktop

## 2. Database Setup (Docker)
Run the following command to start Elasticsearch in the background:

```bash
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" docker.elastic.co/elasticsearch/elasticsearch:8.11.3
```

## 3. Backend Setup
Navigate to the backend directory and set up the Python environment:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Ensure your `.env` file is properly configured with your Google API Key and Data Paths.

## 4. Build FAISS Index (First time only)
Before starting the server, build the FAISS vector index from the CLIP embeddings:

```bash
cd backend
venv\Scripts\activate
python -m app.scripts.build_index
```

## 5. Run the Backend
Start the FastAPI backend server:

```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
The API documentation will be available at: http://localhost:8000/docs

## 6. Frontend Setup and Run
Open a new terminal window, navigate to the frontend directory, and start the React app:

```bash
cd frontend
npm install
npm run dev
```
The Web UI will be available at: http://localhost:3000
