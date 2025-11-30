Crypto Time Series Forecasting Dashboard – ARIMA + LSTM

A professional cryptocurrency price analysis and forecasting dashboard built using Streamlit, Python, ARIMA, and LSTM Deep Learning.
The system loads historical crypto data, visualizes trends, calculates volatility, and predicts future prices using statistical and neural network models.

📌 Features

Interactive Streamlit Dashboard

ARIMA Forecasting (traditional time-series model)

LSTM Forecasting (deep learning sequence model)

Historical Trend Visualization (Plotly)

Volatility & Daily Returns Analysis

Custom Forecast Horizon

Adjustable LSTM window size & epochs

Latest Price, Avg Return, Volatility KPIs

Modern UI + Smooth animations

🧠 Models Used
ARIMA (Statistical Model)

Identifies linear time dependencies

Good for structured, stable trends

Fast and interpretable

LSTM (Deep Learning Model)

RNN architecture designed for sequential data

Captures long-term patterns

Handles crypto volatility effectively

🎮 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/<Samarthechanur/crypto-time-series-dashboard.git

2️⃣ Move into project folder
cd crypto-time-series-dashboard

3️⃣ Create a virtual environment
python -m venv .venv

Activate (Windows):
.venv\Scripts\activate

Activate (Mac/Linux):
source .venv/bin/activate

4️⃣ Install dependencies
pip install -r requirements.txt

5️⃣ Run the dashboard
streamlit run app.py


The app will open at:

http://localhost:8501

📂 Project Structure
crypto-time-series-dashboard/
│-- app.py                      # Main dashboard
│-- requirements.txt            # Dependencies
│-- .gitignore
│-- data/
│    └── Crypto Historical Data.csv
│-- README.md

🛠 Technologies Used

Python 3

Streamlit

Plotly

Pandas & NumPy

Statsmodels (ARIMA)

TensorFlow/Keras (LSTM)

Scikit-Learn

🚀 Deployment (Optional)


👨‍💻 Author

Samarth H – GitHub Profile
