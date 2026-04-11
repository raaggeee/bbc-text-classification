from fastapi import FastAPI
from pydantic import BaseModel
import pickle

class UserRequest(BaseModel):
    user_text: str

app = FastAPI()

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    return model

def load_vectorizer():
    with open("vectorizer.pkl", "rb") as f:
        vect = pickle.load(f)

    return vect

def load_label():
    with open("label.pkl", "rb") as f:
        label = pickle.load(f)

    return label

@app.post("/predict")
def predict(UserRequest):
    model = load_model()
    vectorizer = load_vectorizer()
    label = load_label()

    user_text = vectorizer.transform([UserRequest])
    pred = model.predict(user_text)
    output = label.inverse_transform(pred)
    return {"message", output[0]}