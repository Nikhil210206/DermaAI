// The logic (talks to backend)
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadContainer = document.getElementById('upload-container');

    uploadContainer.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            console.log('File selected:', file.name);
            // TODO: Implement upload to backend
        }
    });
});
