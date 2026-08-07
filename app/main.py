from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.api.analytics import router as analytics_router

# Import database initialization functions
from app.database.sqlite_db import ManifestDB
from app.database.chroma_db import ChromaDBClient
from app.database.analytics_db import AnalyticsDB

# Create FastAPI app
app = FastAPI(
    title="Financial Document Intelligence System",
    description="Incremental RAG-based Financial Document Intelligence System",
    version="1.0.0"
)

# Allow Streamlit frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize databases when FastAPI starts
@app.on_event("startup")
def startup():

    ManifestDB()

    ChromaDBClient()

    AnalyticsDB()

    print("SQLite Initialized")

    print("ChromaDB Initialized")

    print("Analytics Database Initialized")


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Financial Document Intelligence System API is running."
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "backend": "FastAPI",
        "database": "SQLite",
        "vector_db": "ChromaDB"
    }


# Register upload routes
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(analytics_router)