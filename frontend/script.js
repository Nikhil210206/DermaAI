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
    resultContainer.scrollIntoView({ behavior: 'smooth' });

    // 1. Main Prediction
    diseaseName.textContent = data.disease;
    
    const percentage = (data.confidence * 100).toFixed(1) + "%";
    confidenceScore.textContent = percentage;
    confidenceBar.style.width = percentage;

    // Color logic
    confidenceBar.className = 'h-2.5 rounded-full transition-all duration-500'; // Reset classes
    if (data.confidence > 0.85) {
        confidenceBar.classList.add('bg-green-500'); // High confidence
    } else if (data.confidence > 0.60) {
        confidenceBar.classList.add('bg-blue-500');  // Medium confidence
    } else {
        confidenceBar.classList.add('bg-yellow-500'); // Low confidence (Check alternatives)
    }

    // 2. Show Alternatives (New Feature)
    // Check if we already have an alternatives list, if not create it
    let altList = document.getElementById('alt-list');
    if (!altList) {
        const altContainer = document.createElement('div');
        altContainer.className = "mt-4 pt-4 border-t border-slate-100";
        altContainer.innerHTML = `<p class="text-xs text-slate-400 mb-2">Other possibilities:</p><ul id="alt-list" class="space-y-1 text-sm text-slate-600"></ul>`;
        
        // Insert before the disclaimer
        const disclaimer = resultContainer.querySelector('.bg-yellow-50');
        resultContainer.insertBefore(altContainer, disclaimer);
        altList = document.getElementById('alt-list');
    }
    
    // Populate Alternatives
    altList.innerHTML = '';
    if (data.alternatives && data.alternatives.length > 0) {
        data.alternatives.forEach(alt => {
            const li = document.createElement('li');
            li.innerHTML = `<span class="font-medium">${alt.disease}</span> <span class="text-slate-400">(${alt.probability})</span>`;
            altList.appendChild(li);
        });
    } else {
        altList.innerHTML = '<li class="text-slate-400 italic">No other likely matches.</li>';
    }
}