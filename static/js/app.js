// Global variables
let currentForecastData = null;
let forecastChart = null;

// DOM elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileStats = document.getElementById('file-stats');
const removeFileBtn = document.getElementById('remove-file');
const forecastForm = document.getElementById('forecast-form');
const generateBtn = document.getElementById('generate-forecast');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingMessage = document.getElementById('loading-message');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    checkSystemStatus();
});

function initializeEventListeners() {
    // File upload events
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    removeFileBtn.addEventListener('click', removeFile);
    
    // Form events
    forecastForm.addEventListener('submit', handleForecastSubmit);
    
    // Chart controls
    document.getElementById('chart-item').addEventListener('change', updateChart);
    
    // Export button
    document.getElementById('export-csv').addEventListener('click', exportForecast);
}

function checkSystemStatus() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            const statusElement = document.getElementById('system-status');
            if (data.status === 'healthy') {
                statusElement.innerHTML = '<i class="fas fa-circle status-indicator"></i> Ready';
                statusElement.style.color = '#48bb78';
            } else {
                statusElement.innerHTML = '<i class="fas fa-circle status-indicator" style="color: #f56565;"></i> Error';
                statusElement.style.color = '#f56565';
            }
        })
        .catch(error => {
            console.error('System status check failed:', error);
            showToast('System Status', 'Unable to check system status', 'warning');
        });
}

// File upload handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('Invalid File', 'Please select a CSV file', 'error');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) { // 16MB limit
        showToast('File Too Large', 'File size must be less than 16MB', 'error');
        return;
    }
    
    uploadFile(file);
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading('Uploading and validating file...');
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            showFileInfo(data);
            enableForecastGeneration();
            showToast('Upload Successful', `File uploaded with ${data.rows} rows`, 'success');
        } else {
            showToast('Upload Failed', data.error, 'error');
        }
    })
    .catch(error => {
        hideLoading();
        console.error('Upload error:', error);
        showToast('Upload Error', 'Failed to upload file', 'error');
    });
}

function showFileInfo(data) {
    fileName.textContent = data.filename;
    fileStats.textContent = `${data.rows} rows, ${data.columns.length} columns`;
    
    uploadArea.style.display = 'none';
    fileInfo.style.display = 'flex';
    
    // Store filepath for forecast generation
    fileInfo.dataset.filepath = data.filepath;
}

function removeFile() {
    uploadArea.style.display = 'block';
    fileInfo.style.display = 'none';
    fileInput.value = '';
    disableForecastGeneration();
    hideResults();
}

function enableForecastGeneration() {
    generateBtn.disabled = false;
}

function disableForecastGeneration() {
    generateBtn.disabled = true;
}

// Forecast generation
function handleForecastSubmit(e) {
    e.preventDefault();
    
    const filepath = fileInfo.dataset.filepath;
    if (!filepath) {
        showToast('No File', 'Please upload a file first', 'error');
        return;
    }
    
    const formData = new FormData(forecastForm);
    const requestData = {
        filepath: filepath,
        model_type: formData.get('model_type'),
        forecast_days: parseInt(formData.get('forecast_days')),
        anomaly_detection: formData.has('anomaly_detection')
    };
    
    generateForecast(requestData);
}

function generateForecast(requestData) {
    showLoading('Generating forecast...', getLoadingMessage(requestData));
    
    fetch('/api/forecast', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            currentForecastData = data;
            displayResults(data);
            showToast('Forecast Complete', 'Forecast generated successfully', 'success');
        } else {
            showToast('Forecast Failed', data.error, 'error');
        }
    })
    .catch(error => {
        hideLoading();
        console.error('Forecast error:', error);
        showToast('Forecast Error', 'Failed to generate forecast', 'error');
    });
}

function getLoadingMessage(requestData) {
    const messages = [
        'Loading and preprocessing data...',
        'Training machine learning models...',
        'Generating predictions...',
        'Calculating performance metrics...'
    ];
    
    if (requestData.anomaly_detection) {
        messages.push('Running anomaly detection...');
    }
    
    let messageIndex = 0;
    const messageInterval = setInterval(() => {
        if (messageIndex < messages.length) {
            loadingMessage.textContent = messages[messageIndex];
            messageIndex++;
        } else {
            clearInterval(messageInterval);
            loadingMessage.textContent = 'Finalizing results...';
        }
    }, 2000);
    
    return messageInterval;
}

// Results display
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Update model performance
    updateModelPerformance(data.model_performance, data.comparison);
    
    // Update anomaly detection results
    if (data.anomaly_results) {
        updateAnomalyResults(data.anomaly_results);
    } else {
        document.getElementById('anomaly-section').style.display = 'none';
    }
    
    // Update summary statistics
    updateSummaryStats(data.summary);
    
    // Update chart
    updateChart();
    
    // Update table
    updateTable(data.forecast_data);
}

function updateModelPerformance(performance, comparison) {
    document.getElementById('model-name').textContent = performance.model_name || 'Unknown';
    document.getElementById('model-accuracy').textContent = `${performance.accuracy?.toFixed(1) || 'N/A'}%`;
    document.getElementById('model-error').textContent = `±${performance.mae?.toFixed(1) || 'N/A'} units`;
    document.getElementById('model-r2').textContent = performance.r2?.toFixed(3) || 'N/A';
    
    // Show comparison if available
    if (comparison) {
        const comparisonSection = document.getElementById('model-comparison');
        const comparisonText = document.getElementById('comparison-text');
        
        comparisonText.textContent = `${comparison.winner.toUpperCase()} model selected. ` +
            `Regression MAE: ${comparison.regression_mae?.toFixed(1)}, ` +
            `ARIMA MAE: ${comparison.arima_mae?.toFixed(1)}`;
        
        comparisonSection.style.display = 'block';
    }
}

function updateAnomalyResults(anomalyResults) {
    const anomalySection = document.getElementById('anomaly-section');
    
    document.getElementById('total-anomalies').textContent = anomalyResults.total_anomalies || 0;
    document.getElementById('anomaly-rate').textContent = `${anomalyResults.anomaly_percentage || 0}%`;
    
    // Update recent anomalies list
    const anomalyList = document.getElementById('anomaly-list');
    anomalyList.innerHTML = '';
    
    if (anomalyResults.recent_anomalies && anomalyResults.recent_anomalies.length > 0) {
        anomalyResults.recent_anomalies.forEach(anomaly => {
            const item = document.createElement('div');
            item.className = 'anomaly-list-item';
            item.innerHTML = `
                <span>${anomaly.date}</span>
                <span>Inventory: ${anomaly.total_inventory}</span>
                <span>Error: ${anomaly.reconstruction_error.toFixed(4)}</span>
            `;
            anomalyList.appendChild(item);
        });
    } else {
        anomalyList.innerHTML = '<p>No recent anomalies detected</p>';
    }
    
    anomalySection.style.display = 'block';
}

function updateSummaryStats(summary) {
    const summaryGrid = document.getElementById('summary-grid');
    summaryGrid.innerHTML = '';
    
    if (!summary.totals) return;
    
    const items = ['wings', 'tenders', 'fries_reg', 'fries_large', 'veggies', 'dips', 'drinks', 'flavours'];
    const itemNames = {
        'wings': 'Wings',
        'tenders': 'Tenders',
        'fries_reg': 'Regular Fries',
        'fries_large': 'Large Fries',
        'veggies': 'Veggies',
        'dips': 'Dips',
        'drinks': 'Drinks',
        'flavours': 'Flavours'
    };
    
    items.forEach(item => {
        if (summary.totals[item]) {
            const summaryItem = document.createElement('div');
            summaryItem.className = 'summary-item';
            summaryItem.innerHTML = `
                <h4>${itemNames[item]}</h4>
                <div class="total">${summary.totals[item].total_forecast.toLocaleString()}</div>
                <div class="subtitle">Total Forecast</div>
            `;
            summaryGrid.appendChild(summaryItem);
        }
    });
    
    // Add weekend vs weekday comparison if available
    if (summary.weekend_avg && summary.weekday_avg) {
        const comparisonItem = document.createElement('div');
        comparisonItem.className = 'summary-item';
        comparisonItem.innerHTML = `
            <h4>Weekend vs Weekday</h4>
            <div class="total">${((summary.weekend_avg / summary.weekday_avg - 1) * 100).toFixed(1)}%</div>
            <div class="subtitle">Weekend Premium</div>
        `;
        summaryGrid.appendChild(comparisonItem);
    }
}

function updateChart() {
    if (!currentForecastData || !currentForecastData.forecast_data) return;
    
    const selectedItem = document.getElementById('chart-item').value;
    const ctx = document.getElementById('forecast-chart').getContext('2d');
    
    // Destroy existing chart
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    const labels = currentForecastData.forecast_data.map(item => {
        const date = new Date(item.date);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    
    const forecastData = currentForecastData.forecast_data.map(item => item[`${selectedItem}_forecast`]);
    const stockData = currentForecastData.forecast_data.map(item => item[`${selectedItem}_recommended_stock`]);
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Forecast',
                    data: forecastData,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Recommended Stock',
                    data: stockData,
                    borderColor: '#48bb78',
                    backgroundColor: 'rgba(72, 187, 120, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `${selectedItem.charAt(0).toUpperCase() + selectedItem.slice(1).replace('_', ' ')} Forecast`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Quantity'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

function updateTable(forecastData) {
    const tableBody = document.querySelector('#forecast-table tbody');
    tableBody.innerHTML = '';
    
    forecastData.forEach(item => {
        const row = document.createElement('tr');
        if (item.is_weekend) {
            row.classList.add('weekend-row');
        }
        
        const total = (item.wings_forecast || 0) + (item.tenders_forecast || 0) + 
                     (item.fries_reg_forecast || 0) + (item.fries_large_forecast || 0) + 
                     (item.veggies_forecast || 0);
        
        row.innerHTML = `
            <td>${item.date}</td>
            <td>${item.day_of_week}${item.is_weekend ? ' 🌟' : ''}</td>
            <td>${(item.wings_forecast || 0).toLocaleString()}</td>
            <td>${(item.tenders_forecast || 0).toLocaleString()}</td>
            <td>${(item.fries_reg_forecast || 0).toLocaleString()}</td>
            <td>${(item.fries_large_forecast || 0).toLocaleString()}</td>
            <td>${(item.veggies_forecast || 0).toLocaleString()}</td>
            <td>${(item.dips_forecast || 0).toLocaleString()}</td>
            <td>${(item.drinks_forecast || 0).toLocaleString()}</td>
            <td>${(item.flavours_forecast || 0).toLocaleString()}</td>
            <td><strong>${total.toLocaleString()}</strong></td>
        `;
        
        tableBody.appendChild(row);
    });
}

// Export functionality
function exportForecast() {
    if (!currentForecastData || !currentForecastData.forecast_data) {
        showToast('No Data', 'No forecast data to export', 'error');
        return;
    }
    
    fetch('/api/export', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            forecast_data: currentForecastData.forecast_data
        })
    })
    .then(response => {
        if (response.ok) {
            return response.blob();
        }
        throw new Error('Export failed');
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `inventory_forecast_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        showToast('Export Complete', 'Forecast data exported successfully', 'success');
    })
    .catch(error => {
        console.error('Export error:', error);
        showToast('Export Failed', 'Failed to export forecast data', 'error');
    });
}

// Utility functions
function showLoading(title, messageInterval) {
    loadingMessage.textContent = title;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
    currentForecastData = null;
    
    if (forecastChart) {
        forecastChart.destroy();
        forecastChart = null;
    }
}

function showToast(title, message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <h4>${title}</h4>
        <p>${message}</p>
    `;
    
    document.getElementById('toast-container').appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
    
    // Remove on click
    toast.addEventListener('click', () => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    });
}