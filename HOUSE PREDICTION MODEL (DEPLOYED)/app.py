from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('HOUSEDATA.csv')

# Dropping unnecessary columns
df.drop(["society", "balcony", "availability", "area_type"], axis=1, inplace=True)

# Dropping rows with missing values
df = df.dropna(axis=0)

# Converting columns to integer types
df['bath'] = df['bath'].astype(int)
df['price'] = df['price'].astype(int)

# Label encoding for categorical columns
le_location = LabelEncoder()
le_size = LabelEncoder()
le_total_sqft = LabelEncoder()

df['location_encoded'] = le_location.fit_transform(df['location'])  # Fit on the full dataset
df['size_encoded'] = le_size.fit_transform(df['size'])
df['total_sqft_encoded'] = le_total_sqft.fit_transform(df['total_sqft'])

# Create a price category (binary) for logistic regression
median_price = df['price'].median()
df['price_category'] = df['price'].apply(lambda x: 1 if x > median_price else 0)

# Define features and target
X = df[['location_encoded', 'size_encoded', 'total_sqft_encoded', 'bath']]
y = df['price_category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Extract unique values for the dropdown menus
locations = df['location'].unique()  # Original names for display
sizes = df['size'].unique()
total_sqfts = df['total_sqft'].unique()
baths = sorted(df['bath'].unique())  # Sorted for better display

# Route for the home page with dropdown options
@app.route('/')
def home():
    return render_template('home.html', locations=locations, sizes=sizes, total_sqfts=total_sqfts, baths=baths)

# Helper function to handle unseen labels
def safe_encode(label_encoder, value, fallback=0):
    """Safely encode a value, and return a fallback if unseen."""
    try:
        return label_encoder.transform([value])[0]
    except ValueError:
        return fallback  # Default to fallback if label is unseen

# Route to predict based on form input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        location = request.form['location']
        size = request.form['size']
        total_sqft = request.form['total_sqft']
        bath = int(request.form['bath'])

        # Safely encode the selected options using the previously fitted LabelEncoder
        location_encoded = safe_encode(le_location, location)
        size_encoded = safe_encode(le_size, size)
        total_sqft_encoded = safe_encode(le_total_sqft, total_sqft)

        # Input feature array
        input_features = np.array([[location_encoded, size_encoded, total_sqft_encoded, bath]])

        # Predict using the logistic regression model
        prediction = model.predict(input_features)

        # Return prediction (1 = High Price, 0 = Low Price)
        result = 'High' if prediction[0] == 1 else 'Low'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

# to run use:-> gunicorn -w 4 -b 0.0.0.0:8000 app:app