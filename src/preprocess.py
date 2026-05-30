import re


def clean_text(text):
    """
    Basic text cleaning for story generation.
    """
    if text is None:
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def truncate_text(text, max_chars=2000):
    """
    Limit very long stories so training is faster.
    """
    if len(text) <= max_chars:
        return text

    return text[:max_chars].strip()


def format_prompt_story(prompt, story, max_story_chars=2000):
    """
    Format one prompt-story pair into a consistent training example.
    """
    prompt = clean_text(prompt)
    story = clean_text(story)
    story = truncate_text(story, max_chars=max_story_chars)

    return f"Prompt: {prompt}\nStory: {story}\n\n"