import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("data/house_prices.csv")

# Encode categorical variables
categorical_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
                    "airconditioning", "prefarea", "furnishingstatus"]

encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Features and target
X = data.drop("price", axis=1)
y = data["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, "models/model.pkl")
print("✅ Model saved at models/model.pkl")
