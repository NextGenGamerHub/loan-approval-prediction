import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "loanpredict-ai-internship-secure-key-2026")
    
    # Machine Learning Model
    MODEL_PATH = os.getenv("MODEL_PATH", "loan_approval_logistic_regression_model.pkl")
    
    # SQLite Database
    DATABASE_PATH = os.getenv("DATABASE_PATH", "loan_predictions.db")
