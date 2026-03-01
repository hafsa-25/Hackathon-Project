# PalatePal: Your Personal Dish Recommender

PalatePal is a smart dish recommendation system designed to help you discover your next favorite meal. By providing your taste preferences, such as cuisine, spice level, and dietary needs, PalatePal serves up a curated list of dishes tailored just for you.

This project uses a content-based filtering approach, analyzing dish attributes to find the perfect match for your cravings. The interactive web interface is built with Streamlit, allowing for a seamless user experience.

## Features

- **Personalized Recommendations**: Get dish suggestions based on your unique taste profile.
- **Advanced Filtering**: Narrow down your choices by:
  - **Cuisine**: Pakistani, Indian, Italian, or Turkish.
  - **Spice Level**: From 1 (mild) to 5 (very spicy).
  - **Dietary Preference**: Vegetarian or Non-Vegetarian.
  - **Ingredient Avoidance**: Exclude seafood or any other specific ingredients.
- **Interactive Feedback**: Use the "Like" and "Dislike" buttons to refine future recommendations and teach the system your preferences.
- **Detailed Dish Info**: Each recommendation comes with a description, cuisine type, and spice level.

## How It Works

The recommendation engine is built on a content-based filtering model.

1.  **Data Preprocessing (`df.py`)**:
    - A dataset of dishes (`Dataset.csv`) is cleaned and processed.
    - Categorical features like `cuisine` and `dietary_type` are converted into numerical format using One-Hot Encoding.
    - Textual data, specifically `main_ingredients`, is transformed into a numerical feature matrix using TF-IDF (Term Frequency-Inverse Document Frequency).
    - These features, along with the `spice_level`, are combined to create a comprehensive feature vector for each dish.
    - The processed data, encoders, and TF-IDF vectorizer are saved into `model.pkl`.

2.  **Recommendation Logic (`recommender.py`)**:
    - The user's preferences (cuisine, spice level, etc.) are used to create a "user preference vector".
    - The system filters the dataset based on the user's hard constraints (e.g., must be 'Veg', must be 'Indian').
    - **Cosine Similarity** is calculated between the user's preference vector and the feature vectors of the filtered dishes. This measures how "similar" a dish is to the user's taste.
    - The system boosts the similarity score for dishes the user has previously "liked" to enhance personalization.
    - The top 5 dishes with the highest similarity scores are returned as recommendations.

3.  **Web Interface (`app.py`)**:
    - A simple and intuitive UI is created using Streamlit.
    - Users input their preferences via sidebar controls.
    - On clicking "Get Recommendations", the app calls the recommendation engine and displays the resulting dishes.
    - Feedback from "Like" and "Dislike" buttons is used to update the user's taste profile for subsequent recommendations.

## Getting Started

To run PalatePal on your local machine, follow these steps.

### Prerequisites

- Python 3.8 or later
- Pip

### Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hafsa-25/hackathon-project.git
    cd hackathon-project/Project\ PalatePal/
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install streamlit pandas scikit-learn
    ```

3.  **Run the Streamlit application:**
    The `model.pkl` file is already included, so you do not need to run the `df.py` preprocessing script.
    ```bash
    streamlit run app.py
    ```

4.  **Open your browser:**
    Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

5.  **Get Recommendations:**
    Use the sidebar to select your preferences and click the "Get Recommendations" button to see your personalized dish suggestions.

## File Structure

```
└── Project PalatePal/
    ├── Dataset.csv         # The raw dataset with dish information.
    ├── app.py              # The main Streamlit web application.
    ├── df.py               # Script for data preprocessing and model generation.
    ├── model.pkl           # Pre-trained model file with encoders and feature data.
    └── recommender.py      # Core recommendation logic and functions.
