import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS Configuration
# -----------------------------------------------------------------------
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # Allows opening local frontend HTML files directly via file://
]}})

# -----------------------------------------------------------------------
# Load the four persisted model artifacts ONCE at startup
# -----------------------------------------------------------------------
MODEL_DIR = "./model"

try:
    regression_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "lasso_regressor_for_sme_revenue.joblib")
    )
    classification_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib")
    )
    clustering_artifact = joblib.load(
        os.path.join(MODEL_DIR, "kmeans_mall_customer_segmentation.joblib")
    )
    clustering_scaler = clustering_artifact["scaler"]
    clustering_model = clustering_artifact["kmeans"]

    arm_artifact = joblib.load(
        os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
    )
    recommendation_lookup = arm_artifact["recommendation_lookup"]
    sku_lookup = arm_artifact["sku_lookup"]

    print("All four artifacts have been loaded successfully. The API is ready.")
except Exception as e:
    print(f"Error loading model artifacts from disk: {e}")


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "success",
        "message": "Group D9 ML Model API is running.",
        "endpoints": [
            "/api/v1/models/sme-revenue-regressor/predictions",
            "/api/v1/models/sme-credit-risk-classifier/predictions",
            "/api/v1/models/mall-customer-segmenter/predictions",
            "/api/v1/models/ecommerce-recommender/predictions"
        ]
    })


# -----------------------------------------------------------------------
# 1. REGRESSION -- Predict monthly revenue (KES) for an SME
# -----------------------------------------------------------------------
@app.route("/api/v1/models/sme-revenue-regressor/predictions", methods=["POST"])
def predict_revenue():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "Missing or invalid JSON payload."}), 400

    try:
        input_row = pd.DataFrame([payload])
        predicted_log_revenue = regression_pipeline.predict(input_row)[0]
        predicted_revenue_kes = float(np.expm1(predicted_log_revenue))

        return jsonify({
            "predicted_monthly_revenue_kes": round(predicted_revenue_kes, 2),
            "model": "Lasso Regressor",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------------------------------------------------
# 2. CLASSIFICATION -- Predict SME credit risk category (Low/Medium/High)
# -----------------------------------------------------------------------
@app.route("/api/v1/models/sme-credit-risk-classifier/predictions", methods=["POST"])
def predict_credit_risk():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "Missing or invalid JSON payload."}), 400

    try:
        input_row = pd.DataFrame([payload])
        predicted_class = classification_pipeline.predict(input_row)[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
        class_labels = classification_pipeline.named_steps["model"].classes_

        return jsonify({
            "predicted_credit_risk_category": str(predicted_class),
            "class_probabilities": {
                str(label): round(float(prob), 4)
                for label, prob in zip(class_labels, predicted_probabilities)
            },
            "model": "Support Vector Classifier (tuned, SMOTE-resampled at training only)",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------------------------------------------------
# 3. CLUSTERING -- Assign a new mall customer to an existing segment
# -----------------------------------------------------------------------
@app.route("/api/v1/models/mall-customer-segmenter/predictions", methods=["POST"])
def predict_cluster():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    required = ["age", "annual_income_k", "spending_score"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = pd.DataFrame([{
            "age":             float(data["age"]),
            "annual_income_k": float(data["annual_income_k"]),
            "spending_score":  float(data["spending_score"]),
        }])
        cluster = int(clustering_model.predict(clustering_scaler.transform(X.values))[0])
        return jsonify({"cluster": cluster})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

@app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
def recommend_products():
    payload = request.get_json()
    if not payload or ("cart" not in payload and "cart_skus" not in payload):
        return jsonify({"error": "Missing 'cart' list in request payload."}), 400

    try:
        cart = set(payload.get("cart") or payload.get("cart_skus") or [])
        top_n = int(payload.get("top_n", payload.get("n_recommendations", 3)))

        candidates = {}
        for antecedent, consequent_list in recommendation_lookup.items():
            if antecedent.issubset(cart):
                for sku, confidence, lift in consequent_list:
                    if sku not in cart:
                        if sku not in candidates or confidence > candidates[sku][0]:
                            candidates[sku] = (confidence, lift)

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
        recommendations = [
            {
                "sku": str(sku),
                "product_name": sku_lookup.loc[sku, "product_name"],
                "confidence": round(float(confidence), 3),
                "lift": round(float(lift), 2),
            }
            for sku, (confidence, lift) in ranked
        ]

        return jsonify({
            "cart": sorted(cart),
            "recommendations": recommendations,
            "model": "Apriori association rules, cleaned + precomputed lookup",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server
    app.run(debug=True, host="0.0.0.0", port=5000)


    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response