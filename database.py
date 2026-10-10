import os
import sqlite3
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger("LoanPredictAI.Database")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_predictions.db")

def get_connection():
    """Returns a SQLite connection with Row row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite database and tables, and seeds sample records if empty."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reference_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP NOT NULL,
                no_of_dependents INTEGER NOT NULL,
                education TEXT NOT NULL,
                self_employed TEXT NOT NULL,
                income_annum REAL NOT NULL,
                loan_amount REAL NOT NULL,
                loan_term INTEGER NOT NULL,
                cibil_score INTEGER NOT NULL,
                residential_assets_value REAL NOT NULL,
                commercial_assets_value REAL NOT NULL,
                luxury_assets_value REAL NOT NULL,
                bank_asset_value REAL NOT NULL,
                total_assets REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                status TEXT NOT NULL,
                approval_probability REAL NOT NULL,
                rejection_probability REAL NOT NULL
            )
        """)
        conn.commit()

        # Check if database is empty; if so, seed realistic data
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        if count == 0:
            logger.info("Database is empty. Seeding initial realistic loan applicant records...")
            seed_initial_data(conn)
            logger.info("Database initialized and seeded successfully.")

def calculate_model_outcome(dependents, education, self_employed, income, loan_amt, term, cibil, res_asset, comm_asset, lux_asset, bank_asset):
    """
    Computes Logistic Regression logit using the model's authentic learned weights:
    Coefficients:
      loan_id: -0.000408
      no_of_dependents: -0.031985
      income_annum: -5.3358e-7
      loan_amount: 7.5190e-8
      loan_term: -0.189769
      cibil_score: 0.008981
      residential_assets_value: 5.9886e-9
      commercial_assets_value: 8.3535e-9
      luxury_assets_value: 1.1652e-8
      bank_asset_value: 3.0595e-8
      education_Not Graduate: -0.006709
      self_employed_Yes: -0.005515
      intercept: -0.011875
    """
    import math
    edu_not_grad = 1 if education == "Not Graduate" else 0
    self_emp_yes = 1 if self_employed == "Yes" else 0
    loan_id = 1

    logit = (
        -0.01187462778987263
        + (-0.0004082186264781848 * loan_id)
        + (-0.03198513580471353 * dependents)
        + (-5.335823362663281e-07 * income)
        + (7.519048741297579e-08 * loan_amt)
        + (-0.18976926444575554 * term)
        + (0.00898146779734136 * cibil)
        + (5.988613065965513e-09 * res_asset)
        + (8.353492086449328e-09 * comm_asset)
        + (1.1652425867729397e-08 * lux_asset)
        + (3.059519201691126e-08 * bank_asset)
        + (-0.00670949580143896 * edu_not_grad)
        + (-0.005514731184498852 * self_emp_yes)
    )
    # Clip logit to avoid overflow
    logit = max(min(logit, 35.0), -35.0)
    prob_approved = 1.0 / (1.0 + math.exp(-logit))
    prob_rejected = 1.0 - prob_approved

    pred_class = 1 if prob_approved >= 0.50 else 0
    status = "Approved" if pred_class == 1 else "Rejected"

    return pred_class, status, round(prob_approved * 100, 2), round(prob_rejected * 100, 2)

def seed_initial_data(conn):
    """Populates realistic fintech loan records across varying CIBIL scores and asset tiers."""
    cursor = conn.cursor()
    
    samples = [
        # (dep, edu, self_emp, income, loan_amt, term, cibil, res_asset, comm_asset, lux_asset, bank_asset, days_ago, hours_ago)
        (2, "Graduate", "No", 9600000, 29900000, 12, 778, 2400000, 17600000, 22700000, 8000000, 14, 2),
        (0, "Not Graduate", "Yes", 4100000, 12200000, 8, 417, 2700000, 400000, 8800000, 3300000, 13, 5),
        (3, "Graduate", "No", 9100000, 29700000, 20, 506, 7100000, 4500000, 24200000, 12800000, 12, 1),
        (1, "Graduate", "No", 5800000, 15000000, 4, 825, 4500000, 2000000, 11000000, 6200000, 11, 4),
        (0, "Graduate", "Yes", 7200000, 21000000, 6, 760, 5100000, 3200000, 14500000, 7100000, 11, 8),
        (4, "Not Graduate", "Yes", 2400000, 9500000, 18, 380, 1200000, 0, 4500000, 1800000, 10, 3),
        (2, "Graduate", "No", 8500000, 24000000, 10, 710, 6800000, 4100000, 17000000, 8900000, 10, 7),
        (1, "Graduate", "No", 6400000, 18000000, 8, 685, 4200000, 1500000, 12500000, 5800000, 9, 2),
        (2, "Not Graduate", "No", 3200000, 11000000, 16, 440, 1800000, 600000, 5200000, 2100000, 9, 6),
        (0, "Graduate", "No", 12000000, 35000000, 8, 840, 9500000, 8000000, 28000000, 14000000, 8, 3),
        (3, "Graduate", "Yes", 7900000, 22500000, 12, 745, 5900000, 3800000, 16000000, 7800000, 8, 9),
        (1, "Not Graduate", "Yes", 1800000, 8000000, 20, 350, 900000, 0, 3200000, 1100000, 7, 1),
        (2, "Graduate", "No", 5100000, 14000000, 6, 795, 3800000, 1900000, 9800000, 5200000, 7, 5),
        (0, "Graduate", "No", 4500000, 13000000, 10, 630, 3100000, 1200000, 8500000, 4100000, 6, 2),
        (4, "Graduate", "Yes", 6800000, 20000000, 14, 590, 4800000, 2500000, 13000000, 6000000, 6, 8),
        (1, "Graduate", "No", 9800000, 28000000, 6, 810, 7500000, 5500000, 21000000, 10500000, 5, 2),
        (2, "Not Graduate", "No", 2900000, 10500000, 18, 410, 1500000, 400000, 4800000, 1900000, 5, 7),
        (0, "Graduate", "Yes", 8200000, 23000000, 8, 730, 6200000, 3900000, 16500000, 8100000, 4, 1),
        (3, "Graduate", "No", 7400000, 21500000, 10, 670, 5300000, 3100000, 15000000, 7200000, 4, 6),
        (1, "Not Graduate", "Yes", 3600000, 12000000, 14, 460, 2100000, 800000, 6200000, 2500000, 3, 2),
        (0, "Graduate", "No", 10500000, 31000000, 4, 860, 8200000, 6500000, 23000000, 11800000, 3, 5),
        (2, "Graduate", "No", 6200000, 17500000, 12, 690, 4400000, 2200000, 12000000, 5900000, 2, 1),
        (1, "Graduate", "Yes", 5500000, 16000000, 8, 750, 4000000, 2100000, 11200000, 5400000, 2, 4),
        (3, "Not Graduate", "No", 2500000, 9000000, 16, 390, 1300000, 300000, 4100000, 1600000, 2, 8),
        (0, "Graduate", "No", 8900000, 25000000, 6, 815, 6900000, 4600000, 18500000, 9200000, 1, 1),
        (2, "Graduate", "No", 7100000, 20500000, 10, 720, 5200000, 3300000, 14800000, 7000000, 1, 4),
        (1, "Not Graduate", "Yes", 4200000, 13500000, 12, 530, 2600000, 900000, 7500000, 3100000, 1, 7),
        (0, "Graduate", "No", 6700000, 19000000, 8, 765, 4900000, 2800000, 13800000, 6600000, 0, 1),
        (3, "Graduate", "Yes", 9300000, 27000000, 10, 785, 7100000, 5100000, 19500000, 9600000, 0, 3),
        (2, "Not Graduate", "No", 3800000, 12500000, 14, 475, 2300000, 750000, 6700000, 2800000, 0, 5),
    ]

    base_time = datetime.now()

    for idx, s in enumerate(samples, start=1001):
        dep, edu, self_emp, inc, loan_amt, term, cibil, res_asset, comm_asset, lux_asset, bank_asset, days_ago, hours_ago = s
        tot_asset = res_asset + comm_asset + lux_asset + bank_asset
        
        pred_class, status, prob_app, prob_rej = calculate_model_outcome(
            dep, edu, self_emp, inc, loan_amt, term, cibil, res_asset, comm_asset, lux_asset, bank_asset
        )

        record_time = base_time - timedelta(days=days_ago, hours=hours_ago, minutes=random.randint(5, 55))
        ref_id = f"LP-{idx}"

        cursor.execute("""
            INSERT INTO predictions (
                reference_id, created_at, no_of_dependents, education, self_employed,
                income_annum, loan_amount, loan_term, cibil_score,
                residential_assets_value, commercial_assets_value, luxury_assets_value,
                bank_asset_value, total_assets, predicted_class, status,
                approval_probability, rejection_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ref_id, record_time.strftime("%Y-%m-%d %H:%M:%S"), dep, edu, self_emp,
            inc, loan_amt, term, cibil,
            res_asset, comm_asset, lux_asset, bank_asset, tot_asset,
            pred_class, status, prob_app, prob_rej
        ))

    conn.commit()

def save_prediction(data):
    """
    Saves a new loan prediction into SQLite and returns the generated reference ID.
    data: dict containing all applicant attributes and ML results.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Determine next ID
        cursor.execute("SELECT MAX(id) FROM predictions")
        max_id = cursor.fetchone()[0] or 1000
        new_ref_id = f"LP-{max_id + 1}"
        
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tot_asset = (
            float(data.get("residential_assets_value", 0)) +
            float(data.get("commercial_assets_value", 0)) +
            float(data.get("luxury_assets_value", 0)) +
            float(data.get("bank_asset_value", 0))
        )

        cursor.execute("""
            INSERT INTO predictions (
                reference_id, created_at, no_of_dependents, education, self_employed,
                income_annum, loan_amount, loan_term, cibil_score,
                residential_assets_value, commercial_assets_value, luxury_assets_value,
                bank_asset_value, total_assets, predicted_class, status,
                approval_probability, rejection_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            new_ref_id,
            created_at,
            int(data.get("no_of_dependents", 0)),
            str(data.get("education", "Graduate")),
            str(data.get("self_employed", "No")),
            float(data.get("income_annum", 0)),
            float(data.get("loan_amount", 0)),
            int(data.get("loan_term", 1)),
            int(data.get("cibil_score", 650)),
            float(data.get("residential_assets_value", 0)),
            float(data.get("commercial_assets_value", 0)),
            float(data.get("luxury_assets_value", 0)),
            float(data.get("bank_asset_value", 0)),
            tot_asset,
            int(data.get("predicted_class", 0)),
            str(data.get("status", "Rejected")),
            float(data.get("approval_probability", 0.0)),
            float(data.get("rejection_probability", 0.0))
        ))
        conn.commit()
        return new_ref_id

def get_recent_predictions(limit=50, status_filter=None, search=None):
    """Retrieves list of latest prediction records with optional filtering."""
    with get_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if status_filter in ["Approved", "Rejected"]:
            query += " AND status = ?"
            params.append(status_filter)
        
        if search:
            query += " AND (reference_id LIKE ? OR education LIKE ? OR self_employed LIKE ?)"
            s = f"%{search}%"
            params.extend([s, s, s])

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_prediction_by_id(record_id):
    """Retrieves single record by integer ID or reference_id string."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if str(record_id).isdigit():
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (int(record_id),))
        else:
            cursor.execute("SELECT * FROM predictions WHERE reference_id = ?", (str(record_id),))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_dashboard_metrics():
    """
    Computes aggregated summary KPIs and chart analytics for the dashboard.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions")
        rows = [dict(r) for r in cursor.fetchall()]

        total_apps = len(rows)
        if total_apps == 0:
            return {
                "total_apps": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "approval_rate": 0.0,
                "total_loan_volume": 0,
                "avg_cibil": 0,
                "avg_loan_amount": 0,
                "avg_approval_prob": 0.0,
                "charts": {}
            }

        approved_count = sum(1 for r in rows if r["predicted_class"] == 1)
        rejected_count = total_apps - approved_count
        approval_rate = round((approved_count / total_apps) * 100, 1)

        total_loan_volume = sum(r["loan_amount"] for r in rows)
        avg_cibil = round(sum(r["cibil_score"] for r in rows) / total_apps)
        avg_loan_amount = round(total_loan_volume / total_apps)
        avg_approval_prob = round(sum(r["approval_probability"] for r in rows) / total_apps, 1)

        # CIBIL Score brackets analysis
        cibil_brackets = {
            "Poor (300-549)": {"approved": 0, "rejected": 0, "total": 0},
            "Fair (550-649)": {"approved": 0, "rejected": 0, "total": 0},
            "Good (650-749)": {"approved": 0, "rejected": 0, "total": 0},
            "Excellent (750-900)": {"approved": 0, "rejected": 0, "total": 0}
        }
        for r in rows:
            score = r["cibil_score"]
            if score < 550:
                k = "Poor (300-549)"
            elif score < 650:
                k = "Fair (550-649)"
            elif score < 750:
                k = "Good (650-749)"
            else:
                k = "Excellent (750-900)"
            
            cibil_brackets[k]["total"] += 1
            if r["predicted_class"] == 1:
                cibil_brackets[k]["approved"] += 1
            else:
                cibil_brackets[k]["rejected"] += 1

        # Loan term brackets analysis
        terms = {
            "2 - 6 Years": 0,
            "7 - 10 Years": 0,
            "11 - 15 Years": 0,
            "16 - 20+ Years": 0
        }
        for r in rows:
            t = r["loan_term"]
            if t <= 6:
                terms["2 - 6 Years"] += 1
            elif t <= 10:
                terms["7 - 10 Years"] += 1
            elif t <= 15:
                terms["11 - 15 Years"] += 1
            else:
                terms["16 - 20+ Years"] += 1

        # Asset composition: Approved vs Rejected
        approved_rows = [r for r in rows if r["predicted_class"] == 1]
        rejected_rows = [r for r in rows if r["predicted_class"] == 0]

        def avg_asset(row_list, col):
            return round(sum(r[col] for r in row_list) / len(row_list)) if row_list else 0

        asset_breakdown = {
            "categories": ["Residential", "Commercial", "Luxury", "Bank Assets"],
            "approved": [
                avg_asset(approved_rows, "residential_assets_value"),
                avg_asset(approved_rows, "commercial_assets_value"),
                avg_asset(approved_rows, "luxury_assets_value"),
                avg_asset(approved_rows, "bank_asset_value")
            ],
            "rejected": [
                avg_asset(rejected_rows, "residential_assets_value"),
                avg_asset(rejected_rows, "commercial_assets_value"),
                avg_asset(rejected_rows, "luxury_assets_value"),
                avg_asset(rejected_rows, "bank_asset_value")
            ]
        }

        # Education & Employment segment rates
        graduate_rows = [r for r in rows if r["education"] == "Graduate"]
        non_graduate_rows = [r for r in rows if r["education"] == "Not Graduate"]
        salaried_rows = [r for r in rows if r["self_employed"] == "No"]
        self_employed_rows = [r for r in rows if r["self_employed"] == "Yes"]

        def calc_rate(sub_rows):
            if not sub_rows:
                return 0.0
            return round((sum(1 for r in sub_rows if r["predicted_class"] == 1) / len(sub_rows)) * 100, 1)

        demographics_analysis = {
            "labels": ["Graduate", "Not Graduate", "Salaried", "Self-Employed"],
            "approval_rates": [
                calc_rate(graduate_rows),
                calc_rate(non_graduate_rows),
                calc_rate(salaried_rows),
                calc_rate(self_employed_rows)
            ],
            "counts": [
                len(graduate_rows),
                len(non_graduate_rows),
                len(salaried_rows),
                len(self_employed_rows)
            ]
        }

        # Time series: recent 7 daily buckets (or aggregated dates)
        date_counts = {}
        for r in rows:
            d_str = r["created_at"][:10]
            if d_str not in date_counts:
                date_counts[d_str] = {"approved": 0, "rejected": 0}
            if r["predicted_class"] == 1:
                date_counts[d_str]["approved"] += 1
            else:
                date_counts[d_str]["rejected"] += 1

        sorted_dates = sorted(date_counts.keys())[-7:] if len(date_counts) > 7 else sorted(date_counts.keys())
        time_series = {
            "dates": [datetime.strptime(d, "%Y-%m-%d").strftime("%b %d") for d in sorted_dates],
            "approved": [date_counts[d]["approved"] for d in sorted_dates],
            "rejected": [date_counts[d]["rejected"] for d in sorted_dates]
        }

        return {
            "total_apps": total_apps,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "approval_rate": approval_rate,
            "total_loan_volume": total_loan_volume,
            "avg_cibil": avg_cibil,
            "avg_loan_amount": avg_loan_amount,
            "avg_approval_prob": avg_approval_prob,
            "charts": {
                "cibil_brackets": cibil_brackets,
                "terms": terms,
                "asset_breakdown": asset_breakdown,
                "demographics": demographics_analysis,
                "time_series": time_series
            }
        }

def export_predictions_csv():
    """Generates a CSV string of all stored predictions."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Reference ID", "Evaluation Date", "Status", "Approval Probability (%)", "Rejection Probability (%)",
        "Dependents", "Education", "Self Employed", "Annual Income (INR)", "Loan Amount (INR)",
        "Loan Term (Years)", "CIBIL Score", "Residential Assets (INR)", "Commercial Assets (INR)",
        "Luxury Assets (INR)", "Bank Assets (INR)", "Total Assets (INR)"
    ])

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
        for row in cursor.fetchall():
            writer.writerow([
                row["reference_id"],
                row["created_at"],
                row["status"],
                row["approval_probability"],
                row["rejection_probability"],
                row["no_of_dependents"],
                row["education"],
                row["self_employed"],
                row["income_annum"],
                row["loan_amount"],
                row["loan_term"],
                row["cibil_score"],
                row["residential_assets_value"],
                row["commercial_assets_value"],
                row["luxury_assets_value"],
                row["bank_asset_value"],
                row["total_assets"]
            ])

    return output.getvalue()
