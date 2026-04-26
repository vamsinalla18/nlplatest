import os

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence


def train_tokenizer():

    tokenizer = Tokenizer(
        WordPiece(unk_token="[UNK]")
    )

    tokenizer.normalizer = Sequence([
        NFD(),
        Lowercase(),
        StripAccents()
    ])

    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(

        vocab_size=32000,
        min_frequency=2,

        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
        ]
    )

    files = ["data/corpus.txt"]

    if not os.path.exists(files[0]):

        raise ValueError("Corpus file not found")

    print("Training tokenizer...")

    tokenizer.train(files, trainer)

    os.makedirs("tokenizer", exist_ok=True)

    tokenizer.save("tokenizer/medical_tokenizer.json")

    print("Tokenizer saved")


if __name__ == "__main__":
    train_tokenizer()