# 📈 LSTM Stock Price Predictor

This project implements a stock price prediction model using Long Short-Term Memory (LSTM) neural networks, built with TensorFlow/Keras and presented through an interactive **Streamlit** web app. It is designed to forecast stock prices based on historical data.

## 🚀 Features

- Interactive **Streamlit** UI for stock price visualization and forecasting  
- Uses **Yahoo Finance** data via `yfinance`  
- Preprocessing with scaling and sequence generation  
- Trained **LSTM model** for predicting future closing prices  
- Dynamic plotting of actual vs predicted stock prices

## 🛠️ Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Plotly
- yfinance

## 📂 Project Structure

.
├── lstm_predictor.py     # Main Streamlit app
├── README.txt            # Project documentation

## 🔧 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/lstm-stock-predictor.git
   cd lstm-stock-predictor
   ```

2. **Install Dependencies**

   It's recommended to use a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Or install manually:

   ```bash
   pip install streamlit tensorflow pandas numpy scikit-learn yfinance matplotlib plotly
   ```

3. **Run the App**

   ```bash
   streamlit run lstm_predictor.py
   ```

4. **Access the Web App**

   Open [http://localhost:8501](http://localhost:8501) in your browser.

## 📊 How It Works

- Loads stock data using `yfinance`
- Normalizes and prepares sequences
- Trains an LSTM model on historical data
- Uses the model to forecast and display predictions

## 📌 Notes

- For best performance, train on high-quality, high-volume stock datasets.
- This app is designed for educational and experimental purposes.

## 📃 License

MIT License © 2025 [Your Name]
