import streamlit as st
import joblib
import numpy as np
import base64

# -------- LOAD BACKGROUND IMAGE --------

def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg = get_base64("background.jpeg")

# -------- DESIGN --------

st.markdown(
f"""
<style>

.stApp {{
    background-image: url("data:image/jpeg;base64,{bg}");
    background-size: cover;
    background-position: center;
}}

/* FORCE ALL TEXT TO BLACK */

html, body, [class*="css"] {{
    color: black !important;
}}

.title {{
    font-size:55px;
    text-align:center;
    color:black !important;
    font-family:Poppins;
}}

h1, h2, h3, h4, h5, h6 {{
    color:black !important;
}}

p {{
    color:black !important;
}}

label {{
    color:black !important;
    font-weight:bold;
}}

figcaption {{
    color:black !important;
    font-weight:bold;
}}

textarea {{
    border-radius:15px !important;
    border:2px solid #ff69b4 !important;
    color:black !important;
}}

div.stButton > button {{
    background-color:#ffc0cb;
    color:black !important;
    border-radius:20px;
    padding:10px 20px;
}}

</style>
""",
unsafe_allow_html=True
)

# -------- LOAD MODEL AND VECTORIZER --------

model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------- TITLE --------

st.markdown('<h1 class="title">Fake Product Review Detection</h1>', unsafe_allow_html=True)

st.markdown("---")

# -------- PRODUCT IMAGES --------

col1, col2, col3 = st.columns(3)

with col1:
    st.image("rare beauty.jpeg", caption="Rare Beauty", width=150)

with col2:
    st.image("rhode.jpeg", caption="Rhode Beauty", width=150)

with col3:
    st.image("rhode1.jpeg", caption="Rhode", width=120)

st.markdown("---")

# -------- USER INPUT --------

review = st.text_area("Write a beauty product review")

# -------- PREDICTION --------

if st.button("Check Review"):

    if review.strip() == "":
        st.warning("Please write a review first.")

    else:
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("This review looks REAL")
        else:
            st.error("This review looks FAKE")