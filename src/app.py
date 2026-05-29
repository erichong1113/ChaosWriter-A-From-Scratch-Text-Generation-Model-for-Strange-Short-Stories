import os
import torch
import streamlit as st

from model import LSTMStoryModel
from generate import generate_text


MODEL_PATH = "chaoswriter_lstm.pt"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    model_config = checkpoint.get("config", {})

    model = LSTMStoryModel(
        vocab_size=len(stoi),
        embed_dim=model_config.get("embed_dim", 128),
        hidden_dim=model_config.get("hidden_dim", 256),
        num_layers=model_config.get("num_layers", 2)
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, stoi, itos


st.set_page_config(
    page_title="ChaosWriter",
    page_icon="✍️",
    layout="centered"
)

st.title("ChaosWriter")
st.write("A small from-scratch story generation model.")

model, stoi, itos = load_model()

if model is None:
    st.warning(
        "No trained model found. Please train the model first by running:\n\n"
        "`python src/train.py`"
    )
else:
    prompt = st.text_area(
        "Enter a story prompt:",
        value="A student discovers that the vending machine can predict the future.",
        height=100
    )

    max_chars = st.slider(
        "Story length",
        min_value=100,
        max_value=1200,
        value=500,
        step=100
    )

    temperature = st.slider(
        "Creativity / randomness",
        min_value=0.2,
        max_value=1.5,
        value=0.8,
        step=0.1
    )

    if st.button("Generate Story"):
        if not prompt.strip():
            st.error("Please enter a prompt first.")
        else:
            formatted_prompt = f"Prompt: {prompt}\nStory:"

            with st.spinner("Generating story..."):
                story = generate_text(
                    model=model,
                    start_text=formatted_prompt,
                    stoi=stoi,
                    itos=itos,
                    max_new_chars=max_chars,
                    temperature=temperature
                )

            st.subheader("Generated Story")
            st.write(story)