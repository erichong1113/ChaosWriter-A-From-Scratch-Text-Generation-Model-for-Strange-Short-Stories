import streamlit as st

import config
from checkpoint import load_story_model
from generate import generate_text
from runtime import set_random_seed


@st.cache_resource
def load_model():
    try:
        model, stoi, itos, _ = load_story_model(config.BEST_MODEL_PATH)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        return None, None, None, str(error)

    return model, stoi, itos, None


st.set_page_config(
    page_title="ChaosWriter",
    page_icon="✍️",
    layout="centered"
)

st.title("ChaosWriter")
st.caption("A from-scratch character-level LSTM chatbot for strange short stories.")

model, stoi, itos, model_error = load_model()

if model is None:
    st.warning("No usable trained model found. Run `python src/train.py` first.")
    if model_error:
        st.error(model_error)
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Give me a strange story prompt and I will continue it.",
            }
        ]

    with st.sidebar:
        st.header("Generation")

        max_chars = st.slider(
            "Story length",
            min_value=100,
            max_value=1200,
            value=500,
            step=100,
        )

        temperature = st.slider(
            "Creativity",
            min_value=0.2,
            max_value=1.5,
            value=0.8,
            step=0.1,
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            value=config.RANDOM_SEED,
            step=1,
        )

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Enter a strange story prompt")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        formatted_prompt = f"Prompt: {prompt}\nStory:"

        with st.chat_message("assistant"):
            with st.spinner("Generating story..."):
                set_random_seed(int(seed))
                story = generate_text(
                    model=model,
                    start_text=formatted_prompt,
                    stoi=stoi,
                    itos=itos,
                    max_new_chars=max_chars,
                    temperature=temperature,
                )
            st.write(story)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": story,
            }
        )
