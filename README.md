# 🤖 Simple QA Chatbot using PyTorch & Streamlit

This project is a **Question–Answering (QA) chatbot** built using **PyTorch (RNN)** and deployed with **Streamlit**.  
It takes a user question as input and predicts an answer based on a trained dataset.

---

## 🚀 Features

- Custom QA dataset (`CSV`)
- Text preprocessing & vocabulary building
- RNN-based neural network (PyTorch)
- Model & vocabulary persistence
- Interactive Streamlit web app
- Beginner-friendly & interview-ready project

---

## 🧠 Model Architecture

- **Embedding Layer** (50-dim)
- **Simple RNN**
- **Fully Connected Output Layer**
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam

---

## 📂 Project Structure
```bash
project_folder/
│
├── train.py # Model training & saving
├── app.py # Streamlit app
├── qa_rnn_model.pth # Trained model weights
├── vocab.pkl # Vocabulary file
├── QA_Dataset.csv # Question–Answer dataset
└── README.md
```


---

## 📊 Dataset Format (`QA_Dataset.csv`)

```csv
question,answer
What is AI?,Artificial intelligence
What is the capital of France?,Paris
```

---

## ⚙️ Installation
```bash
pip install torch pandas streamlit
```

---

## 🏋️ Train the Model

Run the training script:
```bash
python train.py
```
This will generate:
- qa_rnn_model.pth
- vocab.pkl

---

## 🌐 Run Streamlit App
```bash
streamlit run app.py
```
Open browser at:
```bash
http://localhost:8501
```

---

## 🧪 Example Questions

- What is the capital of France?
- What is AI?
- What is the largest planet?

---

## ⚠️ Limitations

- Predicts single-word answers
- Works best on small datasets
- Simple RNN (not Transformer-based)
> This project is designed for learning, demos, and interviews, not production.

---

## 🔮 Future Improvements

- Multi-word answer generation
- LSTM / GRU upgrade
- Transformer-based QA
- FastAPI backend
- Confidence score display
- Chat-style UI

---

## 👨‍💻 Author

Chintan Dabhi
MCA (AI & ML) Student
Aspiring AI / ML Engineer
