import os
import sys
import logging
from datetime import datetime

import json
from flask import Flask, render_template, request, flash, jsonify, Response, redirect, url_for
import joblib
import pandas as pd
import numpy as np

from config import Config
import database

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LoanPredictAI")

# Initialize Flask App
app = Flask(__name__)
app.config.from_object(Config)

# Initialize SQLite Database & Sample Data
try:
    database.init_db()
    logger.info("Database initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing database: {e}")

# Template Filters
@app.template_filter("format_currency")
def format_currency_filter(val):
    try:
        val = float(val)
        if val >= 10000000:
            return f"₹ {val / 10000000:.2f} Cr"
        elif val >= 100000:
            return f"₹ {val / 100000:.2f} L"
        else:
            return f"₹ {val:,.0f}"
    except (ValueError, TypeError):
        return f"₹ {val}"

@app.template_filter("format_inr")
def format_inr_filter(val):
    try:
        val = float(val)
        return f"₹ {val:,.2f}"
    except (ValueError, TypeError):
        return str(val)

# -------------------------------------------------------------
# MACHINE LEARNING MODEL INITIALIZATION
# -------------------------------------------------------------
model = None
EXPECTED_FEATURES = [
    'loan_id',
    'no_of_dependents',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value',
    'education_Not Graduate',
    'self_employed_Yes'
]

try:
    if os.path.exists(Config.MODEL_PATH):
        model = joblib.load(Config.MODEL_PATH)
        logger.info(f"ML Model loaded successfully from {Config.MODEL_PATH}")
        if hasattr(model, "feature_names_in_"):
            logger.info(f"Model feature_names_in_: {list(model.feature_names_in_)}")
            EXPECTED_FEATURES = list(model.feature_names_in_)
        if hasattr(model, "n_features_in_"):
            logger.info(f"Model n_features_in_: {model.n_features_in_}")
    else:
        logger.error(f"Model file not found at: {Config.MODEL_PATH}")
except Exception as e:
    logger.exception(f"Fatal error loading model from {Config.MODEL_PATH}: {e}")
    model = None

# -------------------------------------------------------------
# ROUTES: LOAN APPROVAL PREDICTION
# -------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Direct endpoint for applicant assessment.
    GET: Displays the loan assessment form directly without authentication.
    POST: Processes applicant metrics with the trained Logistic Regression model.
    """
    if model is None:
        flash("Machine Learning model is currently unavailable on the server. Please verify the model file.", "danger")
        return render_template("prediction.html", form_data={})

    if request.method == "POST":
        errors = []

        # Extract and validate form values
        try:
            no_of_dependents = int(request.form.get("no_of_dependents", -1))
            if no_of_dependents < 0:
                errors.append("Number of dependents must be 0 or greater.")
        except (ValueError, TypeError):
            errors.append("Invalid number of dependents entered.")

        education = request.form.get("education", "").strip()
        if education not in ["Graduate", "Not Graduate"]:
            errors.append("Education must be either 'Graduate' or 'Not Graduate'.")

        self_employed = request.form.get("self_employed", "").strip()
        if self_employed not in ["Yes", "No"]:
            errors.append("Self Employed must be either 'Yes' or 'No'.")

        try:
            income_annum = float(request.form.get("income_annum", -1))
            if income_annum < 0:
                errors.append("Annual income cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Invalid annual income value.")

        try:
            loan_amount = float(request.form.get("loan_amount", -1))
            if loan_amount <= 0:
                errors.append("Loan amount must be greater than zero.")
        except (ValueError, TypeError):
            errors.append("Invalid loan amount value.")

        try:
            loan_term = int(request.form.get("loan_term", -1))
            if loan_term <= 0 or loan_term > 50:
                errors.append("Loan term must be a realistic positive number of years (1–50).")
        except (ValueError, TypeError):
            errors.append("Invalid loan term value.")

        try:
            cibil_score = int(request.form.get("cibil_score", -1))
            if cibil_score < 300 or cibil_score > 900:
                errors.append("CIBIL credit score must be between 300 and 900.")
        except (ValueError, TypeError):
            errors.append("Invalid CIBIL credit score.")

        try:
            residential_assets = float(request.form.get("residential_assets_value", -1))
            if residential_assets < 0:
                errors.append("Residential assets value cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Invalid residential assets value.")

        try:
            commercial_assets = float(request.form.get("commercial_assets_value", -1))
            if commercial_assets < 0:
                errors.append("Commercial assets value cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Invalid commercial assets value.")

        try:
            luxury_assets = float(request.form.get("luxury_assets_value", -1))
            if luxury_assets < 0:
                errors.append("Luxury assets value cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Invalid luxury assets value.")

        try:
            bank_asset = float(request.form.get("bank_asset_value", -1))
            if bank_asset < 0:
                errors.append("Bank asset value cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Invalid bank asset value.")

        # If validation failed, display messages and preserve user input
        if errors:
            for err in errors:
                flash(err, "danger")
            return render_template("prediction.html", form_data=request.form)

        # -------------------------------------------------------------
        # EXACT MODEL PREPROCESSING REPRODUCTION
        # -------------------------------------------------------------
        # Categorical mappings:
        # Graduate -> education_Not Graduate = 0
        # Not Graduate -> education_Not Graduate = 1
        education_not_grad = 1 if education == "Not Graduate" else 0

        # Self Employed:
        # No -> self_employed_Yes = 0
        # Yes -> self_employed_Yes = 1
        self_employed_yes = 1 if self_employed == "Yes" else 0

        # loan_id is an identifier in the model feature list; set sequence identifier
        loan_id = 1

        # Construct DataFrame with exact column names and types
        features_dict = {
            'loan_id': loan_id,
            'no_of_dependents': no_of_dependents,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets,
            'commercial_assets_value': commercial_assets,
            'luxury_assets_value': luxury_assets,
            'bank_asset_value': bank_asset,
            'education_Not Graduate': education_not_grad,
            'self_employed_Yes': self_employed_yes
        }

        input_df = pd.DataFrame([features_dict])

        # Enforce exact column order matching model.feature_names_in_
        input_df = input_df[EXPECTED_FEATURES]

        # -------------------------------------------------------------
        # PREDICTION USING TRAINED LOGISTIC REGRESSION MODEL
        # -------------------------------------------------------------
        try:
            # Classification: 1 = Approved, 0 = Rejected
            prediction_class = int(model.predict(input_df)[0])

            # Prediction Probabilities:
            # Index 0: Rejection Probability
            # Index 1: Approval Probability
            probabilities = model.predict_proba(input_df)[0]
            rejection_probability = float(probabilities[0])
            approval_probability = float(probabilities[1])

            # Standard 0.50 Threshold check
            classification = "Approved" if approval_probability >= 0.50 else "Rejected"
            predicted_class_int = 1 if approval_probability >= 0.50 else 0

            logger.info(
                f"Prediction performed: Approval={approval_probability:.4f}, "
                f"Rejection={rejection_probability:.4f}, Outcome={classification}"
            )

            # Persist prediction in SQLite database for Dashboard & Auditing
            prediction_ref_id = "LP-LIVE"
            try:
                prediction_record = {
                    "no_of_dependents": no_of_dependents,
                    "education": education,
                    "self_employed": self_employed,
                    "income_annum": income_annum,
                    "loan_amount": loan_amount,
                    "loan_term": loan_term,
                    "cibil_score": cibil_score,
                    "residential_assets_value": residential_assets,
                    "commercial_assets_value": commercial_assets,
                    "luxury_assets_value": luxury_assets,
                    "bank_asset_value": bank_asset,
                    "predicted_class": predicted_class_int,
                    "status": classification,
                    "approval_probability": round(approval_probability * 100, 2),
                    "rejection_probability": round(rejection_probability * 100, 2)
                }
                prediction_ref_id = database.save_prediction(prediction_record)
                logger.info(f"Saved prediction record with Reference ID: {prediction_ref_id}")
            except Exception as db_err:
                logger.warning(f"Failed to persist prediction record to database: {db_err}")

        except Exception as e:
            logger.exception(f"Error executing model prediction: {e}")
            flash(f"Error during model prediction: {str(e)}", "danger")
            return render_template("prediction.html", form_data=request.form)

        # Format applicant data dictionary for rendering summary
        applicant_summary = {
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": f"{income_annum:,.2f}",
            "loan_amount": f"{loan_amount:,.2f}",
            "loan_term": f"{loan_term} years",
            "cibil_score": cibil_score,
            "residential_assets_value": f"{residential_assets:,.2f}",
            "commercial_assets_value": f"{commercial_assets:,.2f}",
            "luxury_assets_value": f"{luxury_assets:,.2f}",
            "bank_asset_value": f"{bank_asset:,.2f}"
        }

        return render_template(
            "result.html",
            classification=classification,
            predicted_class=predicted_class_int,
            approval_probability=round(approval_probability * 100, 2),
            rejection_probability=round(rejection_probability * 100, 2),
            threshold="50.00%",
            applicant=applicant_summary,
            prediction_id=prediction_ref_id,
            timestamp=datetime.now().strftime("%B %d, %Y - %I:%M %p")
        )

    return render_template("prediction.html", form_data={})

# -------------------------------------------------------------
# REDIRECT LEGACY DASHBOARD ROUTE
# -------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    """Redirects legacy dashboard URL directly to loan prediction page."""
    return redirect(url_for("predict"))

@app.route("/api/predictions/<reference_id>", methods=["GET"])
def api_prediction_detail(reference_id):
    """Returns detailed financial breakdown of a specific applicant for modal view."""
    record = database.get_prediction_by_id(reference_id)
    if not record:
        return jsonify({"status": "error", "message": "Record not found"}), 404
    return jsonify({"status": "success", "applicant": record})

# -------------------------------------------------------------
# API ROUTE (FOR AUTOMATED TESTING & PROGRAMMATIC ACCESS)
# -------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"status": "error", "message": "ML model is not loaded"}), 500

    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({"status": "error", "message": "No JSON or form data provided"}), 400

    try:
        no_of_dependents = int(data.get("no_of_dependents", 0))
        education = data.get("education", "Graduate")
        self_employed = data.get("self_employed", "No")
        income_annum = float(data.get("income_annum", 0))
        loan_amount = float(data.get("loan_amount", 0))
        loan_term = int(data.get("loan_term", 1))
        cibil_score = int(data.get("cibil_score", 650))
        residential_assets = float(data.get("residential_assets_value", 0))
        commercial_assets = float(data.get("commercial_assets_value", 0))
        luxury_assets = float(data.get("luxury_assets_value", 0))
        bank_asset = float(data.get("bank_asset_value", 0))

        # Preprocessing
        education_not_grad = 1 if education == "Not Graduate" else 0
        self_employed_yes = 1 if self_employed == "Yes" else 0

        input_df = pd.DataFrame([{
            'loan_id': 1,
            'no_of_dependents': no_of_dependents,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets,
            'commercial_assets_value': commercial_assets,
            'luxury_assets_value': luxury_assets,
            'bank_asset_value': bank_asset,
            'education_Not Graduate': education_not_grad,
            'self_employed_Yes': self_employed_yes
        }])[EXPECTED_FEATURES]

        pred = int(model.predict(input_df)[0])
        probs = model.predict_proba(input_df)[0]
        rejection_prob = float(probs[0])
        approval_prob = float(probs[1])
        status_text = "Approved" if approval_prob >= 0.50 else "Rejected"
        pred_class_int = 1 if approval_prob >= 0.50 else 0

        # Persist to database
        ref_id = "API-PRED"
        try:
            ref_id = database.save_prediction({
                "no_of_dependents": no_of_dependents,
                "education": education,
                "self_employed": self_employed,
                "income_annum": income_annum,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil_score,
                "residential_assets_value": residential_assets,
                "commercial_assets_value": commercial_assets,
                "luxury_assets_value": luxury_assets,
                "bank_asset_value": bank_asset,
                "predicted_class": pred_class_int,
                "status": status_text,
                "approval_probability": round(approval_prob * 100, 2),
                "rejection_probability": round(rejection_prob * 100, 2)
            })
        except Exception as db_e:
            logger.warning(f"API database persistence warning: {db_e}")

        return jsonify({
            "status": "success",
            "reference_id": ref_id,
            "prediction": status_text,
            "predicted_class": pred_class_int,
            "approval_probability": round(approval_prob * 100, 2),
            "rejection_probability": round(rejection_prob * 100, 2),
            "threshold": 0.50
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# -------------------------------------------------------------
# ERROR HANDLERS
# -------------------------------------------------------------
@app.errorhandler(404)
def not_found_error(error):
    return render_template("error.html", code=404, message="The requested page could not be found."), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", code=500, message="An internal server error occurred."), 500

# -------------------------------------------------------------
# APPLICATION ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting LoanPredict AI Web Application on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
