// --- DOM ELEMENTS ---
const fileInput = document.getElementById('file-input');
const cameraBtn = document.getElementById('camera-btn');
const closeCameraBtn = document.getElementById('close-camera');
const captureBtn = document.getElementById('capture-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const clearPreviewBtn = document.getElementById('clear-preview');

const cameraSection = document.getElementById('camera-section');
const previewSection = document.getElementById('preview-section');
const resultCard = document.getElementById('result-card');
const loading = document.getElementById('loading');

const videoFeed = document.getElementById('video-feed');
const canvas = document.getElementById('canvas');
const imageDisplay = document.getElementById('image-display');

const resultTitle = document.getElementById('result-title');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const alternativesContainer = document.getElementById('alternatives-container');
const alternativesList = document.getElementById('alternatives-list');

let stream = null;
let currentFile = null;

// --- 1. FILE UPLOAD HANDLER ---
fileInput.addEventListener('change', function(e) {
    if (e.target.files && e.target.files[0]) {
        console.log("File selected:", e.target.files[0].name);
        currentFile = e.target.files[0];
        showPreview(currentFile);
        stopCamera();
    }
});

// --- 2. CAMERA HANDLERS ---
cameraBtn.addEventListener('click', async () => {
    console.log("Opening camera...");
    resetUI();
    cameraSection.classList.remove('hidden');
    
    try {
        // Ask for rear camera on mobile
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: "environment" } 
        });
        videoFeed.srcObject = stream;
    } catch (err) {
        console.error("Camera Error:", err);
        alert("Camera access denied or not available.");
        cameraSection.classList.add('hidden');
    }
});

closeCameraBtn.addEventListener('click', stopCamera);

captureBtn.addEventListener('click', () => {
    // Capture frame to canvas
    const context = canvas.getContext('2d');
    canvas.width = videoFeed.videoWidth;
    canvas.height = videoFeed.videoHeight;
    context.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);

    // Convert to file
    canvas.toBlob((blob) => {
        currentFile = new File([blob], "camera_snap.jpg", { type: "image/jpeg" });
        console.log("Photo captured");
        showPreview(currentFile);
        stopCamera();
    }, 'image/jpeg');
});

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    cameraSection.classList.add('hidden');
}

// --- 3. PREVIEW LOGIC ---
function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imageDisplay.src = e.target.result;
        previewSection.classList.remove('hidden');
        analyzeBtn.classList.remove('hidden');
        
        // Scroll to analyze button
        analyzeBtn.scrollIntoView({ behavior: 'smooth' });
    }
    reader.readAsDataURL(file);
}

clearPreviewBtn.addEventListener('click', resetUI);

function resetUI() {
    currentFile = null;
    fileInput.value = "";
    previewSection.classList.add('hidden');
    resultCard.classList.add('hidden');
    analyzeBtn.classList.add('hidden');
    stopCamera();
}

// --- 4. API CALL (AI ANALYSIS) ---
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // UI State: Loading
    analyzeBtn.classList.add('hidden');
    loading.classList.remove('hidden');
    resultCard.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        // ⚠️ CHANGE URL IF DEPLOYED
        // Local: 'http://127.0.0.1:8000/predict'
        // Render: 'https://your-app-name.onrender.com/predict'
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Backend Error");
        
        const data = await response.json();
        console.log("AI Result:", data);
        displayResult(data);

    } catch (error) {
        console.error(error);
        alert("Server Error: Ensure Backend is running (uvicorn app:app)");
        analyzeBtn.classList.remove('hidden');
    } finally {
        loading.classList.add('hidden');
    }
});

// --- 5. DISPLAY RESULTS ---
function displayResult(data) {
    resultCard.classList.remove('hidden');
    resultCard.scrollIntoView({ behavior: 'smooth' });

    // Title
    resultTitle.textContent = data.disease;
    
    // Confidence Bar
    const percent = Math.round(data.confidence * 100);
    confidenceText.textContent = percent + "%";
    confidenceBar.style.width = percent + "%";

    // Color Coding
    confidenceBar.className = "h-2.5 rounded-full transition-all duration-500"; // Reset
    if (percent > 80) confidenceBar.classList.add("bg-green-500");
    else if (percent > 50) confidenceBar.classList.add("bg-blue-500");
    else confidenceBar.classList.add("bg-yellow-500");

    // Alternatives
    alternativesList.innerHTML = "";
    if (data.alternatives && data.alternatives.length > 0) {
        alternativesContainer.classList.remove('hidden');
        data.alternatives.forEach(alt => {
            const li = document.createElement('li');
            li.className = "flex justify-between";
            li.innerHTML = `<span>${alt.disease}</span> <span class="text-slate-400">${alt.probability}</span>`;
            alternativesList.appendChild(li);
        });
    } else {
        alternativesContainer.classList.add('hidden');
    }
}