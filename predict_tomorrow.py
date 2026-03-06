import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\Santosh\mlProject\sensex_data_clean.csv"
MODEL_FILE = "model_data/sensex_model.keras" 
SCALER_FILE = "model_data/price_scaler.pkl"
WINDOW_SIZE = 60

def predict_future():
    print("[INFO] Waking up the AI Brain...")
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except Exception as e:
        print(f"[ERROR] Missing Model or Scaler. Check 'model_data' folder. Error: {e}")
        return

    print("[INFO] Fetching the latest market data...")
    df = pd.read_csv(INPUT_FILE)
    last_60_days = df['Close'].tail(WINDOW_SIZE).values
    last_date = df['Date'].iloc[-1]
    last_price = last_60_days[-1]
    
    # PREPARE THE DATA FOR THE AI
   
    # 1. Scale it down to decimals (0 to 1)
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    X_input = np.array([last_60_days_scaled])
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    # ASK THE AI
    
    print("[INFO] Analyzing sequence and predicting tomorrow's momentum...")
    predicted_decimal = model.predict(X_input, verbose=0)
    predicted_price = scaler.inverse_transform(predicted_decimal)[0][0]

    # DISPLAY RESULTS
    
    print("\n" + "="*50)
    print(" 🚀 SENSEX AI: FUTURE PREDICTION")
    print("="*50)
    print(f"Last Known Data Date : {last_date}")
    print(f"Last Known Close Price : Rs. {last_price:,.2f}")
    print("-" * 50)
    print(f"🔮 PREDICTED PRICE FOR NEXT TRADING DAY: Rs. {predicted_price:,.2f}")
    print("="*50)

    if predicted_price > last_price:
        print("📈 AI Sentiment: BULLISH (Market expected to rise)")
    else:
        print("📉 AI Sentiment: BEARISH (Market expected to fall)")

if __name__ == "__main__":
    predict_tomorrow = predict_future()


