import io

import sentencepiece as spm


class CharTokenizer:
    tokenizer_type = "char"

    def __init__(self, text=None, stoi=None, itos=None):
        if text is not None:
            chars = sorted(set(text))
            self.stoi = {char: index for index, char in enumerate(chars)}
            self.itos = {index: char for char, index in self.stoi.items()}
        else:
            self.stoi = dict(stoi)
            self.itos = dict(itos)

    @property
    def vocab_size(self):
        return len(self.stoi)

    @classmethod
    def from_mappings(cls, stoi, itos):
        return cls(stoi=stoi, itos=itos)

    def encode(self, text):
        return [self.stoi[char] for char in text if char in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[index] for index in ids)

    def checkpoint_data(self):
        return {
            "tokenizer_type": self.tokenizer_type,
            "stoi": self.stoi,
            "itos": self.itos,
        }


class SentencePieceTokenizer:
    tokenizer_type = "sentencepiece"

    def __init__(self, model_proto):
        self.model_proto = bytes(model_proto)
        self.processor = spm.SentencePieceProcessor(
            model_proto=self.model_proto
        )

    @classmethod
    def train(cls, text, vocab_size):
        model_writer = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(text.splitlines()),
            model_writer=model_writer,
            model_type="bpe",
            vocab_size=vocab_size,
            character_coverage=1.0,
            byte_fallback=True,
            minloglevel=2,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
        )
        return cls(model_writer.getvalue())

    @property
    def vocab_size(self):
        return self.processor.vocab_size()

    def encode(self, text):
        return self.processor.encode(text, out_type=int)

    def decode(self, ids):
        return self.processor.decode([int(index) for index in ids])

    def checkpoint_data(self):
        return {
            "tokenizer_type": self.tokenizer_type,
            "tokenizer_model": self.model_proto,
        }


def tokenizer_from_checkpoint(checkpoint):
    tokenizer_type = checkpoint.get("tokenizer_type", "char")
    if tokenizer_type == "sentencepiece":
        if "tokenizer_model" not in checkpoint:
            raise ValueError("Checkpoint is missing tokenizer_model.")
        return SentencePieceTokenizer(checkpoint["tokenizer_model"])
    if "stoi" not in checkpoint or "itos" not in checkpoint:
        raise ValueError("Checkpoint is missing required keys: itos, stoi")
    return CharTokenizer(stoi=checkpoint["stoi"], itos=checkpoint["itos"])
