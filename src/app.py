import streamlit as st

import config
from checkpoint import load_story_model
from generate import build_anchored_prompt, generate_text
from runtime import set_random_seed


@st.cache_resource
def load_model():
    try:
        model, tokenizer, _ = load_story_model(config.BEST_MODEL_PATH)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        return None, None, str(error)

    return model, tokenizer, None


st.set_page_config(
    page_title="ChaosWriter",
    page_icon="✍️",
    layout="centered"
)

st.title("ChaosWriter")
st.caption("A from-scratch subword LSTM chatbot for strange short stories.")

model, tokenizer, model_error = load_model()

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

        max_tokens = st.slider(
            "Story length",
            min_value=50,
            max_value=400,
            value=200,
            step=50,
        )

        temperature = st.slider(
            "Creativity",
            min_value=0.2,
            max_value=1.5,
            value=0.5,
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

        formatted_prompt = build_anchored_prompt(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating story..."):
                set_random_seed(int(seed))
                generated_text = generate_text(
                    model=model,
                    start_text=formatted_prompt,
                    tokenizer=tokenizer,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                _, marker, continuation = generated_text.partition("Story:")
                story = continuation.strip() if marker else generated_text
            st.write(story)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": story,
            }
        )
