import streamlit as st

import config
from checkpoint import load_story_model
from generate import generate_text


@st.cache_resource
def load_model():
    try:
        model, stoi, itos, _ = load_story_model(config.BEST_MODEL_PATH)
    except FileNotFoundError:
        return None, None, None

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
