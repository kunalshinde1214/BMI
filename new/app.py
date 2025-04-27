from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.svm import SVR  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

app = Flask(__name__)

def load_and_train_model():
    df = pd.read_csv("D:/Projects/BE Project/bmi_predictor/BMI_main.csv")
    df.drop(['Person ID A1'], axis=1, inplace=True)
    df['Height'] = df['Height'] * 100
    df.drop(['Feet', 'Inches', 'Pounds', 'Age', 'BMI'], axis=1, inplace=True)
    
    X = df.drop(['BMI_Post'], axis=1)
    y = df[['BMI_Post']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
    
    rf = RandomForestRegressor()
    rf_params = {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None], "max_leaf_nodes": [10, 30, None]}
    rf_best = RandomizedSearchCV(rf, rf_params, n_iter=5, cv=3, random_state=30, n_jobs=-1)
    rf_best.fit(X_train, y_train.values.ravel())
    
    base_learners = [
        ('svr', make_pipeline(StandardScaler(), SVR(kernel='linear', max_iter=500))),
        ('linear', LinearRegression()),
        ('random_forest', rf_best.best_estimator_),
        ('kneighbors', make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))),
        ('xgboost', XGBRegressor(n_estimators=50, learning_rate=0.1))
    ]
    
    stacking = StackingRegressor(
        estimators=base_learners,
        final_estimator=GradientBoostingRegressor(n_estimators=50)
    )
    
    models = {
        'Support Vector Regressor': SVR(kernel='linear', max_iter=500),
        'K-Neighbors Regressor': KNeighborsRegressor(n_neighbors=20),
        'Decision Tree Regressor': DecisionTreeRegressor(max_leaf_nodes=20),
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': rf_best.best_estimator_,
        'AdaBoost Regressor': AdaBoostRegressor(n_estimators=20),
        'Stacking Regressor': stacking
    }
    
    def evaluate_model(true, predicted):
        mae = mean_absolute_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2 = r2_score(true, predicted)
        return mae, rmse, r2
    
    best_model, best_r2, best_model_name = None, -np.inf, ""
    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        mae, rmse, r2 = evaluate_model(y_test, y_pred)
        print(f"Model: {name} => RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        if r2 > best_r2:
            best_r2, best_model, best_model_name = r2, model, name
    
    print("Best Model:", best_model_name, "with R2:", best_r2)
    return best_model, best_model_name, models

best_model, best_model_name, trained_models = load_and_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    algorithm_options = ['Best Fit'] + list(trained_models.keys())
    return render_template('form.html', algorithm_options=algorithm_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['gender']
        gender_numeric = 1 if gender.lower() == 'male' else 0 if gender.lower() == 'female' else None
        if gender_numeric is None:
            return "Invalid gender provided. Please select 'male' or 'female'."
        
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        selected_algorithm = request.form.get('algorithm', 'Best Fit')
        chosen_model = best_model if selected_algorithm.lower() == "best fit" else trained_models.get(selected_algorithm)
        if chosen_model is None:
            return f"Selected algorithm '{selected_algorithm}' is not available."
        
        input_features = np.array([gender_numeric, height, weight]).reshape(1, -1)
        prediction = chosen_model.predict(input_features)
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
