import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import os
import gspread
from gspread_dataframe import get_as_dataframe
import json

# --- Authenticate with Google Sheets ---
# Get the credentials from the GitHub Secret (environment variable)
creds_json = os.getenv("GCP_SA_KEY")
if not creds_json:
    raise ValueError("GCP_SA_KEY environment variable not set.")

creds_dict = json.loads(creds_json)
gc = gspread.service_account_from_dict(creds_dict)

# --- Load Data from Google Sheet ---
# Replace with your Google Sheet name and worksheet name
spreadsheet_name = "house_price_data" 
worksheet_name = "house_prices"

try:
    spreadsheet = gc.open(spreadsheet_name)
    worksheet = spreadsheet.worksheet(worksheet_name)
    data = get_as_dataframe(worksheet)
    print(f"✅ Successfully loaded {len(data)} rows from Google Sheet.")
except Exception as e:
    print(f"❌ Error loading data from Google Sheet: {e}")
    exit() # Exit if data loading fails

# Drop rows with any missing values that might come from the sheet
data.dropna(inplace=True)

# --- The rest of the training logic remains the same ---
# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Encode categorical variables
categorical_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
                    "airconditioning", "prefarea", "furnishingstatus"]

encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    # Convert column to string to handle mixed types from sheets
    data[col] = encoder.fit_transform(data[col].astype(str))
    encoders[col] = encoder

# Features and target
X = data.drop("price", axis=1)
y = data["price"].astype(float) # Ensure price is numeric

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, "models/model.pkl")
print("✅ Model saved at models/model.pkl")