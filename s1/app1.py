from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.svm import SVR  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import AdaBoostRegressor  
from sklearn.ensemble import StackingRegressor 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

app = Flask(__name__)

def load_and_train_model():
    """
    Loads a predefined CSV file, preprocesses the data, trains several regression models,
    evaluates them, and returns:
      - the best-performing model,
      - the name of the best model,
      - a dictionary of all trained models.
    """
    # Load CSV from a predefined file (ensure BMI_main.csv is in your project root)
    df = pd.read_csv("D:/Projects/BE Project/bmi_predictor/s1/static/BMI_main.csv")
    
    # --- Data Preprocessing ---
    # Remove identifier and unused columns
    df.drop(['Person ID A1'], axis=1, inplace=True)
    # Convert Height from meters to centimeters
    df['Height'] = df['Height'] * 100
    # Retain only the columns needed for prediction: Gender, Height, Weight, BMI_Post
    df.drop(['Feet', 'Inches', 'Pounds', 'Age', 'BMI'], axis=1, inplace=True)
    
    # --- Prepare Features and Target ---
    X = df.drop(['BMI_Post'], axis=1)
    y = df[['BMI_Post']]
    
    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
    
    # --- Define Models ---
    support_vector = SVR(kernel='linear', max_iter=251)
    kneighbors = KNeighborsRegressor(n_neighbors=20)
    decision_tree = DecisionTreeRegressor(max_leaf_nodes=20)
    linear = LinearRegression()
    random_forest = RandomForestRegressor(max_leaf_nodes=30)
    ada_boost = AdaBoostRegressor(n_estimators=20)
    
    # Define a stacking regressor with multiple base learners
    base_learners = [
         ('support_vector', SVR(kernel='linear', max_iter=251)),
         ('linear', LinearRegression()),
         ('random_forest', RandomForestRegressor(max_leaf_nodes=30)),
         ('kneighbors', KNeighborsRegressor(n_neighbors=20))
    ]
    stacking = StackingRegressor(
         estimators=base_learners,
         final_estimator=DecisionTreeRegressor()
    )
    
    # Bundle all models into a dictionary
    models = {
         'Support Vector Regressor': support_vector,
         'K-Neighbors Regressor': kneighbors,
         'Decision Tree Regressor': decision_tree,
         'Linear Regression': linear,
         'Random Forest Regressor': random_forest,
         'AdaBoost Regressor': ada_boost,
         'Stacking Regressor': stacking
    }
    
    # --- Model Evaluation Function ---
    def evaluate_model(true, predicted):
         mae = mean_absolute_error(true, predicted)
         mse = mean_squared_error(true, predicted)
         rmse = np.sqrt(mse)
         r2 = r2_score(true, predicted)
         return mae, rmse, r2
    
    # --- Train Each Model and Select the Best ---
    best_model = None
    best_r2 = -np.inf
    best_model_name = ""
    
    for name, model in models.items():
         # Fit each model (convert y_train to a 1D array to avoid warnings)
         model.fit(X_train, y_train.values.ravel())
         y_pred = model.predict(X_test)
         mae, rmse, r2 = evaluate_model(y_test, y_pred)
         print(f"Model: {name} => RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
         if r2 > best_r2:
             best_r2 = r2
             best_model = model
             best_model_name = name
    
    print("Best Model:", best_model_name, "with R2:", best_r2)
    return best_model, best_model_name, models

# Train the models once when the app starts
best_model, best_model_name, trained_models = load_and_train_model()

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/form')
def form():
    """
    Renders the prediction form page.
    Passes a list of available algorithm options to the template,
    including a 'Best Fit' option which uses the best-performing model.
    """
    # Create a list with 'Best Fit' plus all available algorithm names
    algorithm_options = ['Best Fit'] + list(trained_models.keys())
    return render_template('form.html', algorithm_options=algorithm_options)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Processes the prediction form submission, selects the algorithm based on user input,
    uses the chosen model to predict BMI_Post, and renders the result page.
    """
    try:
        # Retrieve and process form data
        gender = request.form['gender']
        if gender.lower() == 'male':
            gender_numeric = 1
        elif gender.lower() == 'female':
            gender_numeric = 0
        else:
            return "Invalid gender provided. Please select 'male' or 'female'."
        
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        
        # Retrieve the selected algorithm from the form (default to 'Best Fit')
        selected_algorithm = request.form.get('algorithm', 'Best Fit')
        if selected_algorithm.lower() == "best fit":
            chosen_model = best_model
        else:
            chosen_model = trained_models.get(selected_algorithm)
            if chosen_model is None:
                return f"Selected algorithm '{selected_algorithm}' is not available."
        
        # Prepare input features as a (1, 3) NumPy array
        input_features = np.array([gender_numeric, height, weight]).reshape(1, -1)
        prediction = chosen_model.predict(input_features)
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/about', endpoint='about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
