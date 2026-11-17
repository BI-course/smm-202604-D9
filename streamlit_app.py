import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Group D9 ML App", layout="wide")
MODEL_DIR = "./model"

@st.cache_resource
def load_all_artifacts():
    reg = joblib.load(os.path.join(MODEL_DIR, "lasso_regressor_for_sme_revenue.joblib"))
    clf = joblib.load(os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib"))
    clust = joblib.load(os.path.join(MODEL_DIR, "kmeans_mall_customer_segmentation.joblib"))
    arm = joblib.load(os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib"))
    return reg, clf, clust["scaler"], clust["kmeans"], arm["recommendation_lookup"], arm["sku_lookup"]

regressor, classifier, clustering_scaler, clustering_kmeans, recommendation_lookup, sku_lookup = load_all_artifacts()

st.title("Group D9 Machine Learning Model Suite")

tab1, tab2, tab3, tab4 = st.tabs(["Regression", "Classification", "Clustering", "Recommender"])

with tab1:
    st.header("SME Monthly Revenue Predictor")
    c1, c2, c3 = st.columns(3)
    with c1:
        sector = st.text_input("Sector", "Technology")
        county = st.text_input("County", "Nairobi")
        quarter = st.text_input("Quarter", "Q3")
        biz_age = st.number_input("Business Age (Years)", value=4.5)
    with c2:
        employees = st.number_input("Number of Employees", value=12, step=1)
        edu = st.text_input("Owner Education Level", "Secondary")
        marketing = st.number_input("Monthly Marketing Spend (KES)", value=45000.0)
        foot_traffic = st.number_input("Avg Daily Foot Traffic (0 if none)", value=0.0)
    with c3:
        online_score = st.number_input("Online Presence Score", value=78.0)
        loan_amt = st.number_input("Loan Amount (KES)", value=250000.0)
        interest_rate = st.number_input("Interest Rate (%)", value=14.5)
        inflation = st.number_input("Inflation Rate (%)", value=6.5)
        comp_density = st.number_input("Competitor Density", value=5, step=1)
        sat_score = st.number_input("Customer Satisfaction Score", value=4.2)

    if st.button("Predict Revenue"):
        ft_val = None if foot_traffic == 0 else foot_traffic
        df = pd.DataFrame([{
            "sector": sector, "county": county, "quarter": quarter,
            "business_age_years": biz_age, "num_employees": int(employees),
            "owner_education_level": edu, "monthly_marketing_spend_kes": marketing,
            "avg_daily_foot_traffic": ft_val, "online_presence_score": online_score,
            "loan_amount_kes": loan_amt, "interest_rate_pct": interest_rate,
            "inflation_rate_pct": inflation, "competitor_density": int(comp_density),
            "customer_satisfaction_score": sat_score
        }])
        log_rev = regressor.predict(df)[0]
        rev = float(np.expm1(log_rev))
        st.success(f"Predicted Monthly Revenue: KES {rev:,.2f}")

with tab2:
    st.header("SME Credit Risk Classifier")
    col1, col2 = st.columns(2)
    with col1:
        cs = st.text_input("Sector ", "Retail")
        cc = st.text_input("County ", "Nairobi")
        cq = st.text_input("Quarter ", "Q2")
        cp = st.text_input("Loan Purpose", "Working Capital")
        ce = st.text_input("Education Level", "Secondary")
        ca = st.number_input("Business Age", value=2.1)
        cn = st.number_input("Employees", value=4, step=1)
        cm = st.number_input("Monthly Revenue (KES)", value=850000.0)
        cl = st.number_input("Loan Amount", value=150000.0)
    with col2:
        ct = st.number_input("Loan Term (Months)", value=12, step=1)
        ci = st.number_input("Interest Rate (%)", value=16.5)
        col = st.number_input("Collateral Value (0 if none)", value=0.0)
        cdti = st.number_input("Debt-to-Income Ratio", value=0.62)
        ccs = st.number_input("Credit Score", value=590, step=1)
        cpr = st.number_input("Previous Loans", value=1, step=1)
        cmp = st.number_input("Missed Payments", value=3, step=1)
        con = st.number_input("Online Score", value=40.0)
        cst = st.number_input("Satisfaction Score", value=3.4)
        ccd = st.number_input("Competitor Density", value=6, step=1)

    if st.button("Predict Credit Risk"):
        col_val = None if col == 0 else col
        df = pd.DataFrame([{
            "sector": cs, "county": cc, "quarter": cq, "loan_purpose": cp,
            "owner_education_level": ce, "business_age_years": ca, "num_employees": int(cn),
            "monthly_revenue_kes": cm, "loan_amount_kes": cl, "loan_term_months": int(ct),
            "interest_rate_pct": ci, "collateral_value_kes": col_val, "debt_to_income_ratio": cdti,
            "credit_score": int(ccs), "num_previous_loans": int(cpr), "missed_payments_count": int(cmp),
            "online_presence_score": con, "customer_satisfaction_score": cst, "competitor_density": int(ccd)
        }])
        pred = classifier.predict(df)[0]
        probs = classifier.predict_proba(df)[0]
        labels = classifier.named_steps["model"].classes_
        st.success(f"Predicted Credit Risk Category: **{pred}**")
        for l, p in zip(labels, probs):
            st.write(f"- {l}: {p:.4f}")

with tab3:
    st.header("Mall Customer Segmentation (Clustering)")
    age = st.number_input("Age", value=27, step=1)
    income = st.number_input("Annual Income (k$)", value=45.0)
    spending = st.number_input("Spending Score (1-100)", value=72.0, step=1.0)

    if st.button("Assign Cluster"):
        df = pd.DataFrame({"Age": [int(age)], "Annual Income (k$)": [float(income)], "Spending Score (1-100)": [float(spending)]})
        scaled = clustering_scaler.transform(df)
        cluster = int(clustering_kmeans.predict(scaled)[0])
        st.success(f"Assigned Mall Customer Segment (Cluster ID): **{cluster}**")

with tab4:
    st.header("E-Commerce Product Recommender (ARM)")
    cart_input = st.text_input("Enter Cart SKUs (comma-separated)", "H06, H07")
    top_n = st.slider("Number of Recommendations", 1, 5, 3)

    if st.button("Get Recommendations"):
        cart = set([s.strip() for s in cart_input.split(",") if s.strip()])
        candidates = {}
        for antecedent, consequent_list in recommendation_lookup.items():
            if antecedent.issubset(cart):
                for sku, confidence, lift in consequent_list:
                    if sku not in cart:
                        if sku not in candidates or confidence > candidates[sku][0]:
                            candidates[sku] = (confidence, lift)
        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
        if ranked:
            for sku, (conf, lift) in ranked:
                p_name = sku_lookup.loc[sku, "product_name"]
                st.info(f"**SKU:** {sku} | **Product:** {p_name} | **Confidence:** {conf:.3f} | **Lift:** {lift:.2f}")
        else:
            st.warning("No recommendations found for this cart.")