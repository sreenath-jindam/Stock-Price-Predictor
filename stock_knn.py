import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# --- 1) Load and clean the NSE CSV ---
df = pd.read_csv("TATACONSUMER_ALLNSE.csv")
df.columns = df.columns.str.strip()  # remove extra spaces

# Ensure numeric types for price columns
for col in ["OPEN", "HIGH", "LOW", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(subset=["OPEN", "HIGH", "LOW", "close"], inplace=True)

# --- 2) Feature engineering ---
df["Open-Close"] = df["OPEN"] - df["close"]
df["High-Low"] = df["HIGH"] - df["LOW"]

X = df[["Open-Close", "High-Low"]].values

# --- 3) Classification: buy (+1) vs sell (−1) ---
y_cls = np.where(df["close"].shift(-1) > df["close"], 1, -1)[:-1]
X_cls = X[:-1]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=42, shuffle=False
)

params = {"n_neighbors": list(range(2, 16))}
clf = GridSearchCV(KNeighborsClassifier(), params, cv=5)
clf.fit(X_train_c, y_train_c)

print(f"Classification Train Acc: {accuracy_score(y_train_c, clf.predict(X_train_c)):.2f}")
print(f"Classification Test  Acc: {accuracy_score(y_test_c,  clf.predict(X_test_c)):.2f}")

preds_c = clf.predict(X_test_c)
df_cls = pd.DataFrame({
    "Actual": y_test_c,
    "Predicted": preds_c
})
print("\nSample Classification Results:")
print(df_cls.head(10))

# --- 4) Regression: predict closing price ---
y_reg = df["close"].values
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.25, random_state=42, shuffle=False
)

reg = GridSearchCV(KNeighborsRegressor(), params, cv=5)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
print(f"\nRegression RMSE: {rmse:.2f}")

df_reg = pd.DataFrame({
    "Actual Close": y_test_r,
    "Predicted Close": y_pred_r
})
print("\nSample Regression Results:")
print(df_reg.head(10))

# --- 5) Visualize actual vs predicted closing price ---
plt.figure(figsize=(12, 6))
plt.plot(y_test_r, label="Actual Close")
plt.plot(y_pred_r, label="Predicted Close")
plt.title("KNN Regression: Actual vs Predicted Close")
plt.xlabel("Test Sample Index")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
