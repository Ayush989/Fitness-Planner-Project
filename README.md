🏋️ FitGenie - Personalized Gym Recommendation System 💪
🌟 Overview
FitGenie is a Streamlit-based web application designed to deliver personalized gym recommendations using machine learning. By analyzing user inputs like age, height, weight, BMI, fitness goals, and health conditions, it suggests optimal exercises, equipment, and diet plans. The app features a modern UI inspired by ShadCN, an analytics dashboard 📊 for tracking user history, and downloadable reports in Excel format.
✨ Features

Personalized Recommendations 🏃: Predicts exercises, equipment, and diet plans using a RandomForestClassifier wrapped in a MultiOutputClassifier.
Sleek Interface 🎨: Clean, responsive design with custom CSS styling mimicking ShadCN UI.
Analytics Dashboard 📈: Visualize user history with charts and download data as Excel files.
Clustering Insights 🧠: Uses KMeans clustering to group users with similar fitness profiles.
BMI Feedback ⚖️: Provides health insights based on calculated BMI.
Keyword Extraction 🔍: Extracts key terms from fitness goals for enhanced personalization.

🛠️ Tech Stack

Python 🐍: Core programming language.
Streamlit 🌐: Web application framework for the UI.
Pandas & NumPy 📚: Data manipulation and preprocessing.
Scikit-learn 🤖: Machine learning models (RandomForestClassifier, KMeans, LabelEncoder).
Joblib 💾: Model persistence.
XlsxWriter 📄: Excel file generation.
Seaborn 📊: Data visualization for analytics.

🚀 Installation

Clone the repository:git clone https://github.com/yourusername/fitgenie.git
cd fitgenie


Install dependencies:pip install -r requirements.txt


Ensure the dataset gym_recommendation.xlsx is in the project root.
Run the app:streamlit run fitgenie_app.py



🎮 Usage

Home Page 🏠: Enter your age, height, weight, fitness goals, and fitness type to receive tailored recommendations.
Analytics Dashboard 📊: View usage history, visualize BMI trends, and download data as an Excel file.
About Page ℹ️: Learn more about the app's purpose and functionality.

📊 Dataset
The app uses gym_recommendation.xlsx, which includes columns like Age, Height, Weight, BMI, Sex, Hypertension, Diabetes, Level, Fitness Goal, Fitness Type, Exercises, Equipment, and Diet. Ensure this file is available or replace it with a compatible dataset.
🤖 Model

The app trains a MultiOutputClassifier with RandomForestClassifier if no pre-trained model (gym_recommendation_model.pkl) exists.
Preprocessing includes handling missing values, encoding categorical variables, and creating dummy variables for features like Sex and Fitness Goal.

🔮 Future Improvements

🌐 Add support for real-time data updates.
🧠 Integrate additional machine learning models for improved accuracy.
🎨 Enhance the UI with more interactive visualizations.
🌍 Support for multilingual inputs and non-Latin character sets.

