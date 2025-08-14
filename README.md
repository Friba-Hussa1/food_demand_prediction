# 🍗 Restaurant Inventory Forecasting Dashboard  

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](#)  
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](#)  
[![Flask](https://img.shields.io/badge/Flask-2.3+-red)](#)  
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](#)  

A **user-friendly web dashboard** that empowers restaurant managers to **predict inventory needs**,  
**analyze sales trends**, and **detect anomalies** — helping make smarter stocking decisions, all in one place.  

---

## 🎯 **Key Features**

### 📊 Forecasting & Analysis
- **Multiple Models**: Regression, ARIMA, and anomaly detection
- **Model Selection**: Automatically picks the most accurate forecast
- **Anomaly Alerts**: Highlights unusual sales trends in real-time

### 🖥 Manager-Friendly Interface
- **Drag & Drop CSV Upload**
- **Interactive Charts & Tables**
- **Export Forecasts** to CSV
- **Responsive Design** — Mobile & Desktop Friendly

### 🚀 Deployment Ready
- **Dockerized** for quick setup
- **Secure File Handling**
- **Optimized for Performance**
- Works **locally** or in the **cloud**

---

## 📦 **Prerequisites**
- **Python** 3.10+
- **Docker & Docker Compose** (for containerized setup)
- **8GB RAM** (recommended for large datasets)

---

## ⚡ **Quick Start**

### **Option 1 — Run with Docker (Recommended)**
```bash
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
📁 Project Structure
project/
├── app.py                 # Flask backend
├── templates/             # HTML templates
├── static/                # CSS & JS files
├── models/                # Trained ML models
├── uploads/               # User file uploads
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

🛠 How It Works
Upload your sales/inventory CSV
Choose model type & forecast days
Generate forecast with one click
Review charts, tables, and anomalies
Export results to CSV

📊 Example CSV Format
delivery_date,wings,tenders,fries_reg,fries_large,veggies,dips,drinks,flavours
2024-01-01,5139,545,131,145,140,471,217,721
Required columns: delivery_date + product quantities

☁️ Deployment Options
Local Docker
AWS EC2 / Lightsail
Azure Container Instances
Heroku / Render
docker-compose --profile production up -d

👥 Authors
Abdul-Rasaq Omisesan
Bikash Giri
Gavriel Kirichenko
Callum Arul
Friba Hussainyar