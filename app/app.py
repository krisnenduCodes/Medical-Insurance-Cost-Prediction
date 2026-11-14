import streamlit as st
import pandas as pd
import joblib


# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered",
)


# Styling (works in both light and dark themes)
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 780px;}

    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        padding: 2rem 1.5rem;
        border-radius: 18px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
    }
    .hero h1 {margin: 0; font-size: 2rem; color: #ffffff;}
    .hero p  {margin: .5rem 0 0; opacity: .9; font-size: 1rem;}

    .result-card {
        background: linear-gradient(135deg, rgba(15,118,110,.12), rgba(37,99,235,.12));
        border: 1px solid rgba(37, 99, 235, .35);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card .label {font-size: .9rem; opacity: .75; letter-spacing: .05em; text-transform: uppercase;}
    .result-card .value {font-size: 2.8rem; font-weight: 700; margin: .2rem 0;}
    .result-card .sub   {font-size: .95rem; opacity: .75;}

    .badge {
        display: inline-block; padding: .25rem .8rem; border-radius: 999px;
        font-weight: 600; font-size: .85rem; color: #fff;
    }
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.2);
        padding: .8rem 1rem; border-radius: 12px;
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #0f766e, #2563eb);
        color: #fff; border: none; border-radius: 12px;
        padding: .7rem 1rem; font-weight: 600; font-size: 1.05rem;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {filter: brightness(1.1); color: #fff;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/final_insurance_model.pkl")


model = load_model()



# Helper functions
def build_input(age, sex, bmi, children, smoker, region) -> pd.DataFrame:
    """Create the exact feature set the model was trained on."""
    smoker_value = 1 if smoker == "yes" else 0
    return pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
            "age_smoker": [age * smoker_value],
            "bmi_smoker": [bmi * smoker_value],
        }
    )


def predict_cost(*args) -> float:
    return float(model.predict(build_input(*args))[0])


def bmi_category(bmi: float):
    if bmi < 18.5:
        return "Underweight", "#f59e0b"
    if bmi < 25:
        return "Healthy", "#16a34a"
    if bmi < 30:
        return "Overweight", "#f59e0b"
    return "Obese", "#dc2626"


def cost_level(cost: float):
    if cost < 8000:
        return "Low", "#16a34a"
    if cost < 16000:
        return "Moderate", "#f59e0b"
    return "High", "#dc2626"



# Header
st.markdown(
    """
    <div class="hero">
        <h1>🏥 Medical Insurance Cost Prediction</h1>
        <p>Enter your details to get an instant estimate of your annual insurance charges.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Input form (the app only reruns when the button is pressed)
with st.form("prediction_form"):
    st.subheader("👤 Your Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
            help="Body Mass Index = weight (kg) / height (m)²",
        )
        smoker = st.radio("Smoking Status", ["no", "yes"], horizontal=True)

    with col2:
        sex = st.radio("Gender", ["male", "female"], horizontal=True)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        region = st.selectbox(
            "Region", ["southeast", "southwest", "northeast", "northwest"],
            format_func=str.title,
        )

    submitted = st.form_submit_button("🔮 Predict Insurance Cost", use_container_width=True)

# Result
if submitted:
    with st.spinner("Calculating your estimate..."):
        cost = predict_cost(age, sex, bmi, children, smoker, region)

    level, level_color = cost_level(cost)
    bmi_label, bmi_color = bmi_category(bmi)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">Estimated Annual Insurance Cost</div>
            <div class="value">${cost:,.2f}</div>
            <div class="sub">≈ ${cost / 12:,.2f} per month</div>
            <br>
            <span class="badge" style="background:{level_color}">{level} cost profile</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Monthly Cost", f"${cost / 12:,.0f}")
    m2.metric("BMI Category", bmi_label)
    m3.metric("Smoker", "Yes" if smoker == "yes" else "No")

    # What-if insights
    st.subheader("💡 Insights")
    tab1, tab2 = st.tabs(["Impact of smoking", "Your inputs"])

    with tab1:
        if smoker == "yes":
            alt = predict_cost(age, sex, bmi, children, "no", region)
            diff = cost - alt
            st.info(
                f"As a non-smoker, your estimated cost would be **${alt:,.2f}** — "
                f"smoking adds about **${diff:,.2f}** per year."
            )
            st.progress(min(max(alt / cost, 0.0), 1.0), text="Non-smoker cost vs. your cost")
        else:
            alt = predict_cost(age, sex, bmi, children, "yes", region)
            diff = alt - cost
            st.success(
                f"Being a non-smoker saves you about **${diff:,.2f}** per year "
                f"compared with a smoker of the same profile (**${alt:,.2f}**)."
            )
            st.progress(min(max(cost / alt, 0.0), 1.0), text="Your cost vs. smoker cost")

    with tab2:
        st.dataframe(
            build_input(age, sex, bmi, children, smoker, region)[
                ["age", "sex", "bmi", "children", "smoker", "region"]
            ].rename(columns=str.title),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "⚠️ This is an estimate generated by a machine learning model, "
        "not an actual insurance quotation."
    )