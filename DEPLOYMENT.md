# Restaurant Forecasting Tool - GCP Deployment Guide

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed locally
3. **Docker** installed locally
4. **Git** for version control

## Quick Deployment

### 1. Setup GCP Project

```bash
# Create a new project (optional)
gcloud projects create your-project-id --name="Restaurant Forecasting"

# Set the project
gcloud config set project your-project-id

# Enable billing (required for Cloud Run)
# Visit: https://console.cloud.google.com/billing
```

### 2. Deploy to Cloud Run

```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy (replace with your project ID)
./deploy.sh your-project-id
```

### 3. Access Your App

After deployment, you'll get a URL like:
`https://restaurant-forecast-xxxxx-uc.a.run.app`

## Manual Deployment Steps

If you prefer manual deployment:

### 1. Enable APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2. Build Container

```bash
# Build the image
docker build -t gcr.io/your-project-id/restaurant-forecast .

# Push to Container Registry
docker push gcr.io/your-project-id/restaurant-forecast
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy restaurant-forecast \
  --image gcr.io/your-project-id/restaurant-forecast \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --max-instances 10 \
  --port 8080
```

## Local Development

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Direct Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

## Configuration

### Environment Variables

- `FLASK_ENV`: Set to `production` for deployment
- `PORT`: Port number (8080 for Cloud Run)
- `MPLBACKEND`: Set to `Agg` for headless matplotlib

### Resource Limits

- **Memory**: 2GB (adjustable in cloudbuild.yaml)
- **CPU**: 2 vCPU (adjustable in cloudbuild.yaml)
- **Timeout**: 15 minutes (for model training)
- **Max Instances**: 10 (auto-scaling)

## Monitoring

### View Logs

```bash
# Real-time logs
gcloud logs tail --follow \
  --resource-type=cloud_run_revision \
  --resource-labels=service_name=restaurant-forecast

# Recent logs
gcloud logs read \
  --resource-type=cloud_run_revision \
  --resource-labels=service_name=restaurant-forecast \
  --limit=50
```

### Service Status

```bash
# Get service details
gcloud run services describe restaurant-forecast --region=us-central1

# List all revisions
gcloud run revisions list --service=restaurant-forecast --region=us-central1
```

## Updating the Service

### Automatic (Recommended)

```bash
# Re-run deployment script
./deploy.sh your-project-id
```

### Manual

```bash
# Build new image
docker build -t gcr.io/your-project-id/restaurant-forecast:v2 .
docker push gcr.io/your-project-id/restaurant-forecast:v2

# Update service
gcloud run services update restaurant-forecast \
  --image gcr.io/your-project-id/restaurant-forecast:v2 \
  --region us-central1
```

## Cost Optimization

### Cloud Run Pricing

- **CPU**: $0.00002400 per vCPU-second
- **Memory**: $0.00000250 per GB-second
- **Requests**: $0.40 per million requests
- **Free Tier**: 2 million requests, 400,000 GB-seconds, 200,000 vCPU-seconds per month

### Optimization Tips

1. **Auto-scaling**: Service scales to zero when not in use
2. **Resource Limits**: Adjust CPU/memory based on usage
3. **Request Timeout**: Set appropriate timeout for model training
4. **Regional Deployment**: Choose region closest to users

## Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check build logs
   gcloud builds log [BUILD_ID]
   ```

2. **Service Not Starting**
   ```bash
   # Check service logs
   gcloud logs read --resource-type=cloud_run_revision
   ```

3. **Memory Issues**
   ```bash
   # Increase memory in cloudbuild.yaml
   '--memory'
   '4Gi'  # Increase from 2Gi
   ```

4. **Timeout Issues**
   ```bash
   # Increase timeout in cloudbuild.yaml
   '--timeout'
   '1800'  # 30 minutes
   ```

### Health Checks

The service includes health checks at `/health`:

```bash
# Test health endpoint
curl https://your-service-url/health
```

## Security

### Authentication (Optional)

To require authentication:

```bash
# Remove --allow-unauthenticated from deployment
gcloud run services update restaurant-forecast \
  --no-allow-unauthenticated \
  --region us-central1

# Grant access to specific users
gcloud run services add-iam-policy-binding restaurant-forecast \
  --member="user:user@example.com" \
  --role="roles/run.invoker" \
  --region us-central1
```

### HTTPS

Cloud Run automatically provides HTTPS endpoints with managed SSL certificates.

## Support

For issues:
1. Check the logs using the monitoring commands above
2. Verify your GCP project has billing enabled
3. Ensure all required APIs are enabled
4. Check resource quotas in your GCP project

## Next Steps

1. **Custom Domain**: Map a custom domain to your service
2. **CI/CD**: Set up automated deployments with GitHub Actions
3. **Monitoring**: Add Cloud Monitoring and alerting
4. **Database**: Add Cloud SQL for persistent data storage
