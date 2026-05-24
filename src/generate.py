import torch
from model import LSTMStoryModel


def generate_text(model, start_text, stoi, itos, max_new_chars=500, temperature=0.8):
    device = next(model.parameters()).device

    ids = [stoi[ch] for ch in start_text if ch in stoi]
    x = torch.tensor([ids], dtype=torch.long).to(device)

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_chars):
            logits, _ = model(x)

            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_id], dim=1)

    output_ids = x[0].tolist()
    return "".join([itos[i] for i in output_ids])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load("chaoswriter_lstm.pt", map_location=device)

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = LSTMStoryModel(vocab_size=len(stoi)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    prompt = "Prompt: A student finds a notebook that writes back.\nStory:"
    output = generate_text(model, prompt, stoi, itos)

    print(output)


if __name__ == "__main__":
    main()