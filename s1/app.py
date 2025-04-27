from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

app = Flask(__name__)
app.secret_key = 'YourSecretKeyHere'

# -------------------- DATABASE --------------------
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')

init_db()

# -------------------- AUTH --------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email, password, confirm = request.form['email'], request.form['password'], request.form['confirm_password']
        if not email or not password or not confirm:
            flash("All fields required", "danger")
        elif password != confirm:
            flash("Passwords don't match", "danger")
        elif len(password) < 6:
            flash("Password too short", "danger")
        else:
            try:
                hashed = generate_password_hash(password)
                with get_db_connection() as conn:
                    conn.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, hashed))
                    flash("Signup successful! Please login.", "success")
                    return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash("Email already registered", "danger")
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email, password = request.form['email'], request.form['password']
        with get_db_connection() as conn:
            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if user and check_password_hash(user['password'], password):
                session['user'] = email
                flash("Logged in!", "success")
                return redirect(url_for('home'))
            flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for('login'))

# -------------------- BMI PREDICTION --------------------
def evaluate_model(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

def load_and_train_model():
    df = pd.read_csv("D:/Projects/BE Project/bmi_predictor/s1/static/BMI_main.csv")
    df.drop(['Person ID A1', 'Feet', 'Inches', 'Pounds', 'Age', 'BMI'], axis=1, inplace=True)
    df['Height'] *= 100  # Convert height to cm if needed

    X, y = df.drop(columns='BMI_Post'), df['BMI_Post']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    models = {
        'Linear Regression': LinearRegression(),
        'SVR': SVR(kernel='linear', max_iter=251),
        'KNN': KNeighborsRegressor(n_neighbors=20),
        'Decision Tree': DecisionTreeRegressor(max_leaf_nodes=20),
        'Random Forest': RandomForestRegressor(max_leaf_nodes=30),
        'AdaBoost': AdaBoostRegressor(n_estimators=20),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50)
    }

    if xgb_available:
        models['XGBoost'] = XGBRegressor(n_estimators=100, verbosity=0)

    base_learners = [(name, model) for name, model in models.items() if name in ['SVR', 'KNN', 'Random Forest']]
    models['Stacking'] = StackingRegressor(estimators=base_learners, final_estimator=DecisionTreeRegressor())

    # Train all models
    for model in models.values():
        model.fit(X_train, y_train)

    # Scale features for clustering
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Cluster the training data
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train_scaled)

    # Determine the best model for each cluster
    best_model_per_cluster = {}
    for cluster_label in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster_label)[0]
        X_cluster = X_train.iloc[cluster_indices]
        y_cluster = y_train.iloc[cluster_indices]
        best_score = -np.inf
        best_model_for_cluster = None
        for name, model in models.items():
            preds = model.predict(X_cluster)
            score = r2_score(y_cluster, preds)
            if score > best_score:
                best_score = score
                best_model_for_cluster = name
        best_model_per_cluster[cluster_label] = best_model_for_cluster

    # Evaluate overall best model
    best_model, best_score = None, -np.inf
    model_metrics = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        scores = evaluate_model(y_test, preds)
        model_metrics[name] = scores
        if scores['R2'] > best_score:
            best_score, best_model = scores['R2'], model

    return best_model, model_metrics, models, scaler, kmeans, best_model_per_cluster

# Train models globally once
best_model, model_performance, trained_models, scaler, kmeans, best_model_per_cluster = load_and_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    if 'user' not in session:
        flash("Please login to access the form", "warning")
        return redirect(url_for('login'))
    return render_template('form.html', algorithm_options=['Best Fit'] + list(trained_models.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = 1 if request.form['gender'].lower() == 'male' else 0
        height, weight = float(request.form['height']), float(request.form['weight'])
        selected_model = request.form.get('algorithm', 'Best Fit')
        new_input = np.array([[gender, height, weight]])

        if selected_model == 'Best Fit':
            new_input_scaled = scaler.transform(new_input)
            cluster_label = kmeans.predict(new_input_scaled)[0]
            best_model_name = best_model_per_cluster[cluster_label]
            model = trained_models[best_model_name]
        else:
            model = trained_models[selected_model]

        prediction = model.predict(new_input)
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return f"Prediction Error: {str(e)}"

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/model_performance')
def model_performance_route():
    return render_template('performance.html', performances=model_performance)

if __name__ == '__main__':
    app.run(debug=True)