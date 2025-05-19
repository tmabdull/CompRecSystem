# Comparable Properties Recommendation System

A machine learning-based recommendation system that:
- Analyzes a subject property's characteristics
- Evaluates hundreds of nearby candidate properties
- Recommends the top 3 most comparable properties
- Provides clear explanations for why each property was selected

Key features include:
- Standardized data processing to handle inconsistencies in real estate data
- Distance-based filtering to ensure only nearby properties are considered
- Similarity scoring based on multiple property attributes
- Explainable recommendations with specific reasons for each selection
- Simple web interface for property selection and recommendation display

## Quickstart
Clone this repository on your machine before proceeding.

### Requirements
This program requires python3 and several packages: numpy, pandas, scikit-learn, 
py-xgboost, shap, fastapi, uvicorn, python-multipart, and jinja2.

An easy way to install these dependencies is through a 
[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 
virtual env.

If you already have conda configured on your system, 
you can execute the below cmd in your terminal to automatically create a conda env 
called `CompRecEnv` with all of the required dependencies:

```bash
conda create -n CompRecEnv python numpy pandas scikit-learn py-xgboost shap fastapi uvicorn python-multipart jinja2 -c conda-forge
```

Then, you can activate the conda environment with this cmd:

```bash
conda activate CompRecEnv
```

### Starting the Web App
In your terminal, navigate to the project directory. 
Then, run app.py with this cmd:

```bash
python app.py
```

The output should show that your web app is running on localhost 
(don't worry, we'll get off localhost soon :D) port 8000. 

After waiting a few seconds for the property to load, you can 
click that link in the terminal output to open the app on your 
browser.

### Using the Web App
The web app will display a dropdown of subject properties you can select from. 

When you press the "Get Recommendations" button, the app will use the provided 
XGBoost model with SHAP to load the top 3 recommended comparable properties 
along with explanations for why they were selected.

## Future Steps / Improvements
- Modifying the data processing pipeline to parse and include more features 
from the raw data.
- Tuning the XGBoost model for increased performance.
- Feedback/recommendation system where the user can mark whether a recommendation 
is good or bad. The model will then be retrained based on this feedback.
- Prettier frontend
- Getting off localhost :D


