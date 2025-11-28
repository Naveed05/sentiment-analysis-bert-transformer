🚀 Sentiment Analysis with BERT (HuggingFace Transformers + IMDB)

This project demonstrates fine-tuning BERT (Bidirectional Encoder Representations from Transformers) for sentiment classification on the IMDB movie review dataset.
It includes a complete end-to-end pipeline:
🔥 GPU-accelerated training (CUDA / RTX 3050)
📦 Fine-tuning BERT-base-uncased
📊 Evaluation on IMDB dataset
💾 Saving and loading trained model
🌐 FastAPI backend for predictions
🎨 Streamlit UI for real-time sentiment analysis

This is a professional-level NLP project suitable for:
Academic submissions
Portfolio / GitHub showcase
ML/DL internship applications
Production-ready experimentation

📚 Table of Contents
Features
Tech Stack
Project Structure
Setup & Installation
Training the Model
Running the API
Running the Web App
Sample API Request
Model Info

✨ Features

Fine-tunes BERT-base-uncased for binary sentiment classification
Uses HuggingFace Transformers for training
Preprocesses IMDB dataset using datasets library
Fully GPU-accelerated (CUDA 12.1)
Exposes prediction API using FastAPI
Beautiful frontend built on Streamlit
Saves the trained model for deployment

🛠 Tech Stack
Component	Technology
NLP Model	BERT-base-uncased
Training Framework	HuggingFace Transformers
Dataset	IMDB (binary classification)
Backend	FastAPI
Frontend	Streamlit
Runtime	Python 3.10
Hardware	NVIDIA RTX 3050 GPU (CUDA)

⚙️ Setup & Installation
1️⃣ Clone the repository
git clone https://github.com/Naveed05/sentiment-analysis-bert-transformer.git
cd sentiment-analysis-bert-transformer

2️⃣ Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

🧠 Training the Model

Run the training script:

python model/train.py


This will:
Download IMDB dataset
Tokenize inputs
Fine-tune BERT for 3 epochs
Save the model at:
model/bert_imdb_model/

🌐 Running the API (FastAPI)
Start the backend server:
uvicorn api.main:app --reload


API runs at:
👉 http://127.0.0.1:8000

🔥 Sample API Request
POST → /predict/

Request:
{
  "text": "The movie was amazing!"
}
Response:
{
  "prediction": 1
}
0 = Negative
1 = Positive

🎨 Running the Web UI (Streamlit)
streamlit run streamlit_app/app.py


Features:
Text input box
Real-time prediction
Clean UI
Uses FastAPI as backend

🧬 Model Info
Model: bert-base-uncased
Parameters: 110M
Epochs: 3
Batch Size: 8
Optimizer: AdamW

Learning Rate: 2e-5
Labels: 0 = Negative, 1 = Positive

🚀 Future Improvements
Add GPT-based explanation generator
Add confidence scoring
Deploy on HuggingFace Spaces

Add Docker container
Use DistilBERT for faster inference
Add training progress visualization

👤 Author

Mirza Naveed Baig
Deep Learning | NLP | Python | Data Science
GitHub: Naveed05


Future Improvements

Author
