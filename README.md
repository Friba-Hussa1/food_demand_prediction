🍗 Restaurant Inventory Forecasting Dashboard
A user-friendly web dashboard that helps restaurant managers predict inventory needs and plan ahead using proven forecasting techniques.
The system combines historical sales data, time-series trends, and anomaly detection to help you make smarter stocking decisions — all in one place.

🎯 Key Features
📊 Forecasting & Analysis
Multiple Models: Regression, ARIMA, and anomaly detection
Model Selection: Automatically picks the best-performing forecast
Anomaly Alerts: Highlights unusual trends in sales data
🖥 Manager-Friendly Interface
Drag & Drop Uploads (CSV format)
Interactive Charts & Tables
Export Results to CSV
Mobile & Desktop Friendly
🚀 Ready for Deployment
Dockerized for easy setup
Secure File Handling
Optimized for Performance
Works on local machines or in the cloud
📦 Prerequisites
Python 3.10+
Docker & Docker Compose (for containerized setup)
At least 8GB RAM (recommended for larger datasets)
⚡ Quick Start
Option 1 — Run with Docker (Recommended)
git clone <repository-url>
cd restaurant-inventory-forecasting
docker-compose up --build
Access at: http://localhost:5000
Option 2 — Run Locally
git clone <repository-url>
cd restaurant-inventory-forecasting

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start application
python app.py
Access at: http://localhost:5000
📁 Project Overview
project/
├── app.py                 # Flask backend
├── templates/             # HTML templates
├── static/                # CSS & JavaScript
├── models/                # Trained ML models
├── uploads/               # User file uploads
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
📊 How It Works
Upload your sales/inventory CSV file
Select model type & forecast period
Generate forecasts with a single click
Review charts, tables, and anomaly highlights
Export results for operational planningg
☁️ Deployment Options
Local Docker
AWS EC2 / Lightsail
Azure Container Instances
Heroku / Render
Deployment is as simple as:
docker-compose --profile production up -d
🛠 Example CSV Format
delivery_date,wings,tenders,fries_reg,fries_large,veggies,dips,drinks,flavours
2024-01-01,5139,545,131,145,140,471,217,721
Required: delivery_date + product quantity columns
👥 Authors
Abdul-Rasaq Omisesan
Bikash Giri
Gavriel Kirichenko
Callum Arul
Friba Hussainyar