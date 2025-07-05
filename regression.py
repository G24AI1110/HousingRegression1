from utils import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Load and prepare data
df = load_data()
X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results = []

# Linear Regression (no hyperparameters)
lr = LinearRegression()
lr.fit(X_train, y_train)
preds = lr.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
results.append(("Linear Regression (No Tuning)", mse, r2))

# Random Forest with GridSearchCV
rf = RandomForestRegressor(random_state=42)
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, n_jobs=-1, scoring='r2')
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
rf_preds = best_rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)
results.append(("Random Forest (Tuned)", rf_mse, rf_r2))

# SVR with GridSearchCV
svr = SVR()
svr_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}
svr_grid = GridSearchCV(svr, svr_param_grid, cv=3, n_jobs=-1, scoring='r2')
svr_grid.fit(X_train, y_train)
best_svr = svr_grid.best_estimator_
svr_preds = best_svr.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_preds)
svr_r2 = r2_score(y_test, svr_preds)
results.append(("SVR (Tuned)", svr_mse, svr_r2))

# Save Results
df_result = pd.DataFrame(results, columns=["Model", "MSE", "R2"])
print(df_result)
df_result.to_csv("hyperparameter_results.csv", index=False)
