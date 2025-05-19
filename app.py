# app.py
import os
import uvicorn
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from data_processing import load_and_process_data
from rec_system import load_latest_model_and_scaler, recommend_comps

# Initialize FastAPI app
app = FastAPI(title="Property Comp Recommendation System")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Create a basic HTML template for the homepage
with open("templates/index.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Property Comp Recommendation System</title>
    </head>
    <body>
        <h1>Property Comp Recommendation System - Taha Abdullah</h1>
        
        <h2>Select a Subject Property</h2>
        <form method="post">
            <select name="subject_address">
                {% for address in subject_addresses %}
                <option value="{{ address }}">{{ address }}</option>
                {% endfor %}
            </select>
            <button type="submit">Get Recommendations</button>
        </form>
        
        {% if subject %}
        <h2>Subject Property</h2>
        <div>
            <p><strong>Address:</strong> {{ subject.std_full_address }}</p>
            <p><strong>GLA:</strong> {{ subject.gla }} sq ft</p>
            <p><strong>Bedrooms:</strong> {{ subject.bedrooms }}</p>
        </div>
        {% endif %}
        
        {% if recommendations %}
        <h2>Recommended Comparable Properties</h2>
        {% for rec in recommendations %}
        <div>
            <p><strong>Address:</strong> {{ rec.std_full_address }}</p>
            <p><strong>Explanation:</strong></p>
            <ul>
            {% for reason in rec.explanation %}
                <li>{{ reason }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endfor %}
        {% endif %}
    </body>
    </html>
    """)
    # <p><strong>Similarity Score:</strong> {{ rec.comp_score }}</p>

# Load data at startup
try:
    if (os.path.exists("./data/processed/processed_subjects.pkl") and 
        os.path.exists("./data/processed/processed_comps.pkl") and 
        os.path.exists("./data/processed/processed_properties.pkl")):
        # Load directly from pkl
        print("Appraisal df pkl files found! Loading...")
        subjects_df = pd.read_pickle("./data/processed/processed_subjects.pkl")
        comps_df = pd.read_pickle("./data/processed/processed_comps.pkl")
        properties_df = pd.read_pickle("./data/processed/processed_properties.pkl")
    else:
        # Create dfs by processing the raw data
        print("Could not find appraisal df pkl files. Creating dfs by processing the raw data...")
        subjects_df, comps_df, properties_df = load_and_process_data()
    print("Appraisal dfs loaded successfully")
except Exception as e:
    print(f"Error loading appraisal dfs: {e}")
    subjects_df, comps_df, properties_df = None, None, None

# Load the latest model and scaler
try:
    model, scaler = load_latest_model_and_scaler()
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model and scaler: {e}")
    # Initialize with None if loading fails
    model, scaler = None, None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with subject property selection dropdown"""
    # Get unique subject addresses
    subject_addresses = subjects_df['std_full_address'].dropna().unique().tolist()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "subject_addresses": subject_addresses,
        "subject": None,
        "recommendations": None
    })

@app.post("/", response_class=HTMLResponse)
async def get_recommendations(request: Request, subject_address: str = Form(...)):
    """Process form submission and display recommendations"""
    # Get unique subject addresses for the dropdown
    subject_addresses = subjects_df['std_full_address'].dropna().unique().tolist()
    
    # Get the selected subject property
    subject_row = subjects_df[subjects_df['std_full_address'] == subject_address].iloc[0]

    # Convert Series to dictionary to avoid the "truth value of a Series is ambiguous" error
    subject = subject_row.to_dict()

    # Get candidate properties for this subject
    candidates = properties_df[properties_df['subject_address'] == subject['std_full_address']]
    
    # Get recommended comps
    recommendations = recommend_comps(model, scaler, subject, candidates, top_n=3)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "subject_addresses": subject_addresses,
        "subject": subject,
        "recommendations": recommendations.to_dict('records')
    })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
