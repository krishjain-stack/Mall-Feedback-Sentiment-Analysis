# app.py
# -------------------------------
# EB Mall Feedback Sentiment Analysis
# Frontend + Backend Integration
# -------------------------------

# Step 1: Import Libraries
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# -------------------------------
# Step 2: Build Input Form
# -------------------------------
st.title("📝 EB Mall Feedback Sentiment Classifier")
st.write("Enter your details and feedback below:")

with st.form(key='feedback_form'):
    name = st.text_input("Name")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=1, max_value=120)
    email = st.text_input("Email")
    feedback = st.text_area("Your Feedback")
    submit_button = st.form_submit_button(label='Submit')

# -------------------------------
# Step 3: Load Trained Model & Vectorizer
# -------------------------------
with open("log_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# -------------------------------
# Step 4: Make Predictions
# -------------------------------
if submit_button and feedback.strip() != "":
    input_vector = vectorizer.transform([feedback])
    prediction = model.predict(input_vector)
    sentiment = label_encoder.inverse_transform(prediction)[0]

    # Display sentiment with emoji
    if sentiment.lower() == "positive":
        st.success(f"Sentiment: ✅ Positive")
    else:
        st.error(f"Sentiment: ❌ Negative")
    
    # Display confidence if available
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_vector).max()
        st.caption(f"Confidence: {prob*100:.1f}%")

# -------------------------------
# Step 5: Generate Reports
# -------------------------------
report_file = "EB mall_feedback.csv"

if submit_button and feedback.strip() != "":
    record = {
        "Name": name,
        "Gender": gender,
        "Age": age,
        "Email": email,
        "Feedback": feedback,
        "Sentiment": sentiment
    }

    if os.path.exists(report_file):
        df_existing = pd.read_csv(report_file)
        df_existing = df_existing.append(record, ignore_index=True)
        df_existing.to_csv(report_file, index=False)
    else:
        df_new = pd.DataFrame([record])
        df_new.to_csv(report_file, index=False)

    st.success("Feedback saved successfully!")

# -------------------------------
# Step 6: Display Feedback Report
# -------------------------------
if st.checkbox("Show Feedback Report"):
    if os.path.exists(report_file):
        df_report = pd.read_csv(report_file)
        st.dataframe(df_report)
    else:
        st.warning("No feedback data available yet.")
# -------------------------------
# Step 7: Visualize Sentiment Distribution
# -------------------------------
if st.checkbox("Show Sentiment Charts"):
    if os.path.exists(report_file):
        df_report = pd.read_csv(report_file)
        sentiment_counts = df_report['Sentiment'].value_counts()

        st.subheader("Sentiment Distribution - Bar Chart")
        st.bar_chart(sentiment_counts)

        st.subheader("Sentiment Distribution - Pie Chart")
        st.write(
            df_report['Sentiment'].value_counts().plot.pie(
                autopct='%1.1f%%', startangle=90, figsize=(5,5)
            )
        )
    else:
        st.warning("No feedback data available to show charts.")