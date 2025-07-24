import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from datetime import datetime
from io import BytesIO

# Apply custom styles to mimic ShadCN UI
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e; color: #ffffff;
        font-family: Arial, sans-serif;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTextInput, .stSelectbox, .stNumberInput {
        border-radius: 8px;
        padding: 10px;
    }
    .stTitle {
        color: #2c3e50;
        text-align: center;
    }
    .stSubheader {
        color: #34495e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Analytics Dashboard"])

# Load dataset
data = pd.read_excel("gym recommendation.xlsx", sheet_name="Sheet1")

def preprocess_data(data):
    for col in ['Age', 'Height', 'Weight', 'BMI']:
        data[col] = data[col].fillna(data[col].median())
    for col in ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']:
        data[col] = data[col].fillna(data[col].mode()[0])

    target_columns = ['Exercises', 'Equipment', 'Diet']
    encoders = {}
    for col in target_columns:
        encoder = LabelEncoder()
        data[col + '_encoded'] = encoder.fit_transform(data[col])
        encoders[col] = encoder

    data = pd.get_dummies(data, columns=['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type'])
    columns_to_drop = ['ID', 'Exercises', 'Equipment', 'Diet', 'Recommendation']
    for col in target_columns:
        columns_to_drop.append(col + '_encoded')

    feature_columns = [col for col in data.columns if col not in columns_to_drop]
    X = data[feature_columns]
    y = data[[col + '_encoded' for col in target_columns]]

    return X, y, encoders, feature_columns

X, y, encoders, feature_columns = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def load_model():
    try:
        return joblib.load("gym_recommendation_model.pkl")
    except FileNotFoundError:
        model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
        model.fit(X_train, y_train)
        joblib.dump(model, "gym_recommendation_model.pkl")
        return model

def predict_gym_recommendation(model, user_input, feature_cols, encoders):
    X_user = pd.DataFrame([user_input])
    X_user = pd.get_dummies(X_user)
    X_user = X_user.reindex(columns=feature_cols, fill_value=0)
    predictions = model.predict(X_user)
    results = {}
    for i, target in enumerate(['Exercises', 'Equipment', 'Diet']):
        results[target] = encoders[target].inverse_transform(predictions[:, i])[0]
    return results

def bmi_feedback(bmi):
    if bmi < 18.5:
        return "Underweight - Consider a balanced diet and strength training."
    elif 18.5 <= bmi < 25:
        return "Normal weight - Maintain with regular fitness routines."
    elif 25 <= bmi < 30:
        return "Overweight - Focus on cardio and diet control."
    else:
        return "Obese - Consider a doctor-approved plan focusing on weight loss."

def extract_keywords(goal_text):
    keywords = re.findall(r"\b\w+\b", goal_text.lower())
    return ', '.join(set(keywords))

def suggest_similar_cluster(user_input, X, feature_cols):
    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_cols, fill_value=0)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    cluster_label = kmeans.predict(df)[0]
    return f"You belong to cluster {cluster_label}, similar to others who succeeded with this plan."

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='History')
    processed_data = output.getvalue()
    return processed_data

# In-memory history log
if 'history' not in st.session_state:
    st.session_state.history = []

if page == "About":
    st.title("About Gym Recommendation App")
    st.write("""
    This app provides personalized gym recommendations based on your age, BMI, fitness goals, and health conditions.
    It utilizes machine learning to suggest the best exercises, equipment, and diet plans to help you achieve your fitness goals.
    """)

elif page == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    if len(st.session_state.history) == 0:
        st.info("No usage data available yet. Run some recommendations first.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)
        st.line_chart(history_df[['BMI']])
        st.bar_chart(history_df['Fitness Goal'].value_counts())

        excel_data = convert_df_to_excel(history_df)
        st.download_button(
            label="📥 Download History as Excel",
            data=excel_data,
            file_name="gym_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.title("Gym Recommendation System")
    st.write("Enter your details to get personalized gym recommendations.")

    # User Inputs
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    bmi = weight / (height ** 2)
    st.markdown(f"**Your BMI:** {bmi:.2f} — {bmi_feedback(bmi)}")

    fitness_goal = st.text_input("Fitness Goal")
    fitness_type = st.text_input("Fitness Type")

    user_info = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "BMI": bmi,
        "Fitness Goal": fitness_goal,
        "Fitness Type": fitness_type
    }

    if st.button("Get Recommendations"):
        model = load_model()
        recommendations = predict_gym_recommendation(model, user_info, feature_columns, encoders)
        st.subheader("Gym Recommendations")
        for key, value in recommendations.items():
            st.markdown(f"""
            <div style='padding:10px; border-radius:10px; background:#333333; color: #ffffff; margin-bottom:10px;'>
                <b>{key}:</b> {value}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"**Goal Keywords:** {extract_keywords(fitness_goal)}")
        st.markdown(suggest_similar_cluster(user_info, X, feature_columns))

        chart_data = pd.DataFrame({
            'Recommendation': list(recommendations.keys()),
            'Value': [1, 1, 1]
        })
        st.bar_chart(chart_data.set_index("Recommendation"))

        # Append session data
        session_entry = user_info.copy()
        session_entry.update(recommendations)
        session_entry['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append(session_entry)
