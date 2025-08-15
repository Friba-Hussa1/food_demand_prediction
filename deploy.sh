#!/bin/bash

# 🍗 Simple Restaurant Forecasting Deployment
# Usage: ./deploy.sh your-project-name

echo "🍗 Deploying Restaurant Forecasting Tool..."
echo "=========================================="

# Get project ID
if [ -z "$1" ]; then
    echo "❌ Please provide your project name:"
    echo "   ./deploy.sh my-restaurant-app"
    exit 1
fi

PROJECT_ID=$1
echo "📋 Using project: $PROJECT_ID"

# Set the project
echo "🔧 Setting up Google Cloud..."
gcloud config set project $PROJECT_ID

# Enable required services
echo "🔧 Enabling required services..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com

# Deploy
echo "🚀 Building and deploying your app..."
gcloud builds submit --config cloudbuild.yaml

# Get the URL
echo "🔍 Getting your app URL..."
SERVICE_URL=$(gcloud run services describe restaurant-forecast --region=us-central1 --format="value(status.url)" 2>/dev/null || echo "")

if [ -n "$SERVICE_URL" ]; then
    echo ""
    echo "✅ SUCCESS! Your app is live at:"
    echo "🌐 $SERVICE_URL"
    echo ""
    echo "📋 What to do next:"
    echo "  1. Visit the URL above"
    echo "  2. Upload a CSV file to test"
    echo "  3. Share the URL with others!"
    echo ""
    echo "💡 To update your app, just run this script again"
else
    echo "⚠️  Deployment completed but couldn't get URL"
    echo "   Check Google Cloud Console: https://console.cloud.google.com"
fi
