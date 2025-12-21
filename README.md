# DermaAI

## Project Structure
- **ml/**: The Brains (Machine Learning)
  - `data/`: [EMPTY] Put HAM10000 images here
  - `model.h5`: The saved brain (created by train_model.py)
  - `train_model.py`: The teacher script
- **backend/**: The Server (FastAPI)
  - `app.py`: Main server file
  - `utils.py`: Image processing helper
  - `requirements.txt`: List of python libraries needed
- **frontend/**: The Face (Website)
  - `index.html`: The structure of the site
  - `styles.css`: (Optional) we used Tailwind via CDN
  - `script.js`: The logic (talks to backend)

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
3. Install backend requirements: `pip install -r backend/requirements.txt`
4. Run backend: `uvicorn backend.app:app --reload`
5. Open `frontend/index.html` in your browser.
