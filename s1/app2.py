from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.svm import SVR  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'Kunalshindesfinalyearprojectbmiprediction'  # Replace this with a secure random string in production

# -------------------- USER AUTHENTICATION --------------------
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if not email or not password or not confirm_password:
            flash('All fields are required.', 'danger')
        elif password != confirm_password:
            flash('Passwords do not match.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            try:
                conn = get_db_connection()
                conn.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, hashed_password))
                conn.commit()
                conn.close()
                flash('Signup successful! Please login.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Email already registered.', 'danger')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user'] = email
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# -------------------- BMI MODEL LOGIC --------------------
def load_and_train_model():
    df = pd.read_csv("D:/Projects/BE Project/bmi_predictor/s1/static/BMI_main.csv")
    df.drop(['Person ID A1'], axis=1, inplace=True)
    df['Height'] = df['Height'] * 100
    df.drop(['Feet', 'Inches', 'Pounds', 'Age', 'BMI'], axis=1, inplace=True)

    X = df.drop(['BMI_Post'], axis=1)
    y = df[['BMI_Post']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    support_vector = SVR(kernel='linear', max_iter=251)
    kneighbors = KNeighborsRegressor(n_neighbors=20)
    decision_tree = DecisionTreeRegressor(max_leaf_nodes=20)
    linear = LinearRegression()
    random_forest = RandomForestRegressor(max_leaf_nodes=30)
    ada_boost = AdaBoostRegressor(n_estimators=20)
    base_learners = [
        ('support_vector', support_vector),
        ('linear', linear),
        ('random_forest', random_forest),
        ('kneighbors', kneighbors)
    ]
    stacking = StackingRegressor(estimators=base_learners, final_estimator=DecisionTreeRegressor())

    models = {
        'Support Vector Regressor': support_vector,
        'K-Neighbors Regressor': kneighbors,
        'Decision Tree Regressor': decision_tree,
        'Linear Regression': linear,
        'Random Forest Regressor': random_forest,
        'AdaBoost Regressor': ada_boost,
        'Stacking Regressor': stacking
    }

    def evaluate_model(true, predicted):
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, predicted)
        return mae, rmse, r2

    best_model = None
    best_r2 = -np.inf
    best_model_name = ""
    for name, model in models.items():
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

best_model, best_model_name, trained_models = load_and_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    if 'user' not in session:
        flash("Login required to access the form.", "warning")
        return redirect(url_for('login'))
    algorithm_options = ['Best Fit'] + list(trained_models.keys())
    return render_template('form.html', algorithm_options=algorithm_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['gender']
        gender_numeric = 1 if gender.lower() == 'male' else 0
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        selected_algorithm = request.form.get('algorithm', 'Best Fit')
        chosen_model = best_model if selected_algorithm.lower() == 'best fit' else trained_models.get(selected_algorithm)

        input_features = np.array([gender_numeric, height, weight]).reshape(1, -1)
        prediction = chosen_model.predict(input_features)
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/about')
def about():
    return render_template('about.html')



if __name__ == '__main__':
    app.run(debug=True)
