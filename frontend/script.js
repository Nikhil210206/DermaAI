const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const uploadPrompt = document.getElementById('upload-prompt');
const analyzeBtn = document.getElementById('analyze-btn');
const loading = document.getElementById('loading');
const resultContainer = document.getElementById('result-container');
const diseaseName = document.getElementById('disease-name');
const confidenceScore = document.getElementById('confidence-score');
const confidenceBar = document.getElementById('confidence-bar');

let selectedFile = null;

// Handle File Selection
fileInput.addEventListener('change', function(e) {
    if (e.target.files && e.target.files[0]) {
        selectedFile = e.target.files[0];
        
        // Show Preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            uploadPrompt.classList.add('hidden');
        }
        reader.readAsDataURL(selectedFile);
    }
});

// Handle Analysis
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert("Please select an image first.");
        return;
    }

    // UI Updates
    analyzeBtn.disabled = true;
    loading.classList.remove('hidden');
    resultContainer.classList.add('hidden');

    // Prepare Data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        // IMPORTANT: Update this URL to your deployed backend URL later
        // For local development, use http://127.0.0.1:8000/predict
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();

        // Display Results
        displayResult(data);

    } catch (error) {
        console.error(error);
        alert("Error analyzing image. Ensure backend is running.");
    } finally {
        analyzeBtn.disabled = false;
        loading.classList.add('hidden');
    }
});

function displayResult(data) {
    resultContainer.classList.remove('hidden');
    
    // Smooth scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth' });

    diseaseName.textContent = data.disease;
    
    // Format confidence percentage
    const percentage = (data.confidence * 100).toFixed(1) + "%";
    confidenceScore.textContent = percentage;
    confidenceBar.style.width = percentage;

    // Color code bar based on confidence
    if (data.confidence > 0.8) {
        confidenceBar.classList.remove('bg-yellow-500', 'bg-red-500');
        confidenceBar.classList.add('bg-blue-600');
    } else if (data.confidence > 0.5) {
        confidenceBar.classList.remove('bg-blue-600', 'bg-red-500');
        confidenceBar.classList.add('bg-yellow-500');
    } else {
        confidenceBar.classList.remove('bg-blue-600', 'bg-yellow-500');
        confidenceBar.classList.add('bg-red-500');
    }
}