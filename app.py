# app.py
# -------------------------------
# EB Mall Feedback Sentiment Analysis
# With Background + Side by Side Charts
# -------------------------------

# Step 1: Import Libraries
import streamlit as st           # For web app interface
import pandas as pd             # For handling CSV files and dataframes
import os                       # For checking if file exists
from textblob import TextBlob   # For sentiment analysis
import matplotlib.pyplot as plt  # For plotting charts
import base64                   # For encoding background image

# -------------------------------
# Background Image CSS
# -------------------------------
def set_background(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{image_file}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Convert local image to base64 so Streamlit can display it
with open("back2.png", "rb") as f:   # <-- using back2.png
    encoded_image = base64.b64encode(f.read()).decode()

set_background(encoded_image)

# -------------------------------
# Step 2: Build Input Form
# -------------------------------
st.title("📝 EB Mall Feedback Sentiment Classifier")
st.write("Enter your details and feedback below:")

with st.form(key='feedback_form'):
    name = st.text_input("Name")  # User name input
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])  # Dropdown for gender
    age = st.number_input("Age", min_value=1, max_value=120)      # Numeric input for age
    email = st.text_input("Email")   # User email input
    feedback = st.text_area("Your Feedback")  # Text area for feedback
    submit_button = st.form_submit_button(label='Submit')  # Submit button

# -------------------------------
# Step 3: Analyze Sentiment Using TextBlob
# -------------------------------
if submit_button and feedback.strip() != "":
    blob = TextBlob(feedback)        # Create TextBlob object
    polarity = blob.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

    # Determine sentiment based on polarity
    if polarity > 0:
        sentiment = "Positive"
        st.success(f"Sentiment: ✅ Positive")
    elif polarity < 0:
        sentiment = "Negative"
        st.error(f"Sentiment: ❌ Negative")
    else:
        sentiment = "Neutral"
        st.info(f"Sentiment: ⚪ Neutral")
    
    st.caption(f"Confidence (polarity score): {polarity:.2f}")  # Show polarity score

# -------------------------------
# Step 4: Generate Reports
# -------------------------------
report_file = "EB mall_feedback.csv"  # CSV file to store feedback

if submit_button and feedback.strip() != "":
    record = {
        "Name": name,
        "Gender": gender,
        "Age": age,
        "Email": email,
        "Feedback": feedback,
        "Sentiment": sentiment
    }

    # Append to existing CSV or create a new one
    if os.path.exists(report_file):
        df_existing = pd.read_csv(report_file)
        df_existing = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
        df_existing.to_csv(report_file, index=False)
    else:
        df_new = pd.DataFrame([record])
        df_new.to_csv(report_file, index=False)

    st.success("Feedback saved successfully!")

# -------------------------------
# Step 5: Display Feedback Report
# -------------------------------
if st.checkbox("Show Feedback Report"):
    if os.path.exists(report_file):
        df_report = pd.read_csv(report_file)
        st.dataframe(df_report)
    else:
        st.warning("No feedback data available yet.")

# -------------------------------
# Step 6: Visualize Sentiment Distribution
# -------------------------------
if st.checkbox("Show Sentiment Charts"):
    if os.path.exists(report_file):
        df_report = pd.read_csv(report_file)
        sentiment_counts = df_report['Sentiment'].value_counts()  # Count Positive, Negative, Neutral

        # Create two columns for charts side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Distribution - Bar Chart")
            st.bar_chart(sentiment_counts)

        with col2:
            st.subheader("Sentiment Distribution - Pie Chart")
            fig, ax = plt.subplots()
            sentiment_counts.plot.pie(
                autopct='%1.1f%%', startangle=90, ax=ax
            )
            st.pyplot(fig)
    else:
        st.warning("No feedback data available to show charts.")
