# -------------------------------
# EB Mall Feedback Sentiment Analysis
# Frontend + Backend Integration
# -------------------------------

# Step 1: Import Libraries
import streamlit as st           # For creating web app interface
import pandas as pd             # For handling CSV and dataframes
import pickle                   # For loading saved ML models and objects
import numpy as np              # For numerical operations (used by ML model)
import os                       # For file operations (check if file exists)
import matplotlib.pyplot as plt  # For plotting pie charts

# -------------------------------
# Step 2: Build Input Form
# -------------------------------
st.title("📝 EB Mall Feedback Sentiment Classifier")  # App title
st.write("Enter your details and feedback below:")    # Short instructions

# Create a form to collect user input
with st.form(key='feedback_form'):
    name = st.text_input("Name")  # User name input
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])  # Dropdown for gender
    age = st.number_input("Age", min_value=1, max_value=120)      # Numeric input for age
    email = st.text_input("Email")   # User email input
    feedback = st.text_area("Your Feedback")  # Text area for feedback
    submit_button = st.form_submit_button(label='Submit')  # Submit button

# -------------------------------
# Step 3: Load Trained Model & Vectorizer
# -------------------------------
# Load the saved machine learning model
with open("log_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the saved TF-IDF vectorizer used for text preprocessing
with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Load the saved label encoder to decode predicted labels
with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# -------------------------------
# Step 4: Make Predictions
# -------------------------------
# Check if form submitted and feedback is not empty
if submit_button and feedback.strip() != "":
    # Convert feedback text to vector using TF-IDF vectorizer
    input_vector = vectorizer.transform([feedback])
    # Predict sentiment using trained ML model
    prediction = model.predict(input_vector)
    # Convert numeric prediction back to label (Positive/Negative)
    sentiment = label_encoder.inverse_transform(prediction)[0]

    # Display sentiment with emoji
    if sentiment.lower() == "positive":
        st.success(f"Sentiment: ✅ Positive")  # Green success box
    else:
        st.error(f"Sentiment: ❌ Negative")    # Red error box

    # If model provides probability scores, show confidence
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_vector).max()  # Get highest probability
        st.caption(f"Confidence: {prob*100:.1f}%")     # Show as percentage

# -------------------------------
# Step 5: Generate Reports
# -------------------------------
report_file = "EB mall_feedback.csv"  # CSV file to store feedback records

# Save feedback to CSV if submitted
if submit_button and feedback.strip() != "":
    record = {
        "Name": name,
        "Gender": gender,
        "Age": age,
        "Email": email,
        "Feedback": feedback,
        "Sentiment": sentiment
    }

    # If CSV exists, append new record
    if os.path.exists(report_file):
        df_existing = pd.read_csv(report_file)
        df_existing = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
        df_existing.to_csv(report_file, index=False)
    else:
        # If CSV does not exist, create new file
        df_new = pd.DataFrame([record])
        df_new.to_csv(report_file, index=False)

    st.success("Feedback saved successfully!")  # Confirmation message

# -------------------------------
# Step 6: Display Feedback Report
# -------------------------------
# Checkbox to show feedback report
if st.checkbox("Show Feedback Report"):
    if os.path.exists(report_file):
        df_report = pd.read_csv(report_file)
        st.dataframe(df_report)  # Show data in table format
    else:
        st.warning("No feedback data available yet.")  # Show warning if CSV not present

# -------------------------------
# Step 7: Visualize Sentiment Distribution
# -------------------------------
# Checkbox to show sentiment charts
if st.checkbox("Show Sentiment Charts"):
    if os.path.exists(report_file):
        df_report = pd.read_csv(report_file)
        sentiment_counts = df_report['Sentiment'].value_counts()  # Count positive vs negative

        # Bar chart
        st.subheader("Sentiment Distribution - Bar Chart")
        st.bar_chart(sentiment_counts)

        # Pie chart
        st.subheader("Sentiment Distribution - Pie Chart")
        fig, ax = plt.subplots()
        sentiment_counts.plot.pie(
            autopct='%1.1f%%', startangle=90, ax=ax  # Pie chart with percentage labels
        )
        st.pyplot(fig)  # Show pie chart in Streamlit
    else:
        st.warning("No feedback data available to show charts.")  # Warning if CSV empty
