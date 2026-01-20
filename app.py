import streamlit as st
import torch
import torch.nn as nn
import pickle


# Model Definition (SAME as training)
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 50)
        self.rnn = nn.RNN(50, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))



# Text Processing
def tokenize(text):
    text = text.lower().replace("?", "").replace("'", "")
    return text.split()

def text_to_indices(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)]



# Load Vocab & Model
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = SimpleRNN(len(vocab))
model.load_state_dict(torch.load("qa_rnn_model.pth", map_location="cpu"))
model.eval()



# Prediction Function-
def predict(question, threshold=0.5):
    indices = text_to_indices(question, vocab)
    if len(indices) == 0:
        return "Please enter a valid question"

    q_tensor = torch.tensor(indices).unsqueeze(0)

    with torch.no_grad():
        output = model(q_tensor)
        probs = torch.softmax(output, dim=1)
        value, index = torch.max(probs, dim=1)

    if value.item() < threshold:
        return "I don't know"

    return list(vocab.keys())[index.item()]



# Streamlit UI
st.set_page_config(page_title="QA Chatbot", page_icon="🤖")

st.title("🤖 Simple QA Chatbot")
st.write("Ask questions based on the trained dataset")

question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please type a question")
    else:
        answer = predict(question)
        st.success(f"Answer: {answer}")
