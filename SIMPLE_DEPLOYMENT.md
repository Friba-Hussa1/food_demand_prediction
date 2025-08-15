# 🚀 Simple Deployment Guide

## What This Does

Puts your restaurant forecasting tool on the internet so anyone can use it!

**Before**: Only works on your computer  
**After**: Works for anyone at `https://your-app-name.run.app`

## One-Time Setup (15 minutes)

### Step 1: Install Google Cloud SDK

**Mac:**
```bash
brew install google-cloud-sdk
```

**Windows:** Download from https://cloud.google.com/sdk/docs/install

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
```

### Step 2: Login to Google Cloud
```bash
gcloud auth login
```

### Step 3: Create a Project
```bash
# Replace "my-restaurant-app" with your preferred name
gcloud projects create my-restaurant-app
gcloud config set project my-restaurant-app
```

### Step 4: Enable Billing
- Go to https://console.cloud.google.com/billing
- Link a credit card (Google gives $300 free credit)

## Deploy Your App (2 commands)

```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy (replace with your project name)
./deploy.sh my-restaurant-app
```

**That's it!** You'll get a URL like: `https://restaurant-forecast-xxxxx.run.app`

## What Users See

1. **Visit your URL** - no installation needed
2. **Upload their CSV file** - drag and drop
3. **Click "Generate Forecast"** - AI does the work
4. **Download results** - CSV export

## Cost

- **Free tier**: 2 million requests per month
- **Typical cost**: $0-5/month for small restaurant use
- **Auto-scaling**: Only pay when someone uses it

## Managing Your App

**View logs:**
```bash
gcloud logs tail --follow --resource-type=cloud_run_revision
```

**Update your app:**
```bash
./deploy.sh my-restaurant-app
```

**Delete your app:**
```bash
gcloud run services delete restaurant-forecast --region=us-central1
```

## Troubleshooting

**Build fails?**
- Check that Docker is installed: `docker --version`
- Make sure you're in the project directory

**Can't access the URL?**
- Wait 2-3 minutes after deployment
- Check the URL in your terminal output

**Need help?**
- Check Google Cloud Console: https://console.cloud.google.com
- View service status in Cloud Run section

## Security Notes

- Your app is public by default (anyone can use it)
- Users can only upload CSV files (safe)
- No user data is permanently stored
- Files are automatically deleted after processing

## What Happens Behind the Scenes

1. **Docker** packages your Python app
2. **Google Cloud Build** creates a container
3. **Cloud Run** hosts your container
4. **Auto-scaling** handles traffic automatically
5. **HTTPS** is provided automatically

Your app becomes a professional web service that can handle multiple users simultaneously!
