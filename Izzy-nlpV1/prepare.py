from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64

class PrepareDataset:
    def __init__(self):
        self.n_sentences = 10000
        self.train_split = 0.9

    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)
        return tokenizer

    def find_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)

    def find_vocab_size(self, tokenizer):
        return len(tokenizer.word_index) + 1

    def __call__(self, filename):
        # Load dataset
        dataset = load_dataset("csv", data_files=filename)["train"]

        # Reduce to n_sentences
        dataset = dataset.select(range(min(self.n_sentences, len(dataset))))

        # Add start/end tokens
        dataset = dataset.map(lambda x: {
            "ewe": "<START> " + x["ewe"] + " <EOS>",
            "fr": "<START> " + x["fr"] + " <EOS>"
        })

        # Shuffle
        dataset = dataset.shuffle(seed=42)

        # Split
        train_size = int(self.train_split * len(dataset))
        train = dataset.select(range(train_size))

        # Encoder
        enc_tokenizer = self.create_tokenizer(train["ewe"])
        en_seq_length = self.find_seq_length(train["ewe"])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer)

        trainX = enc_tokenizer.texts_to_sequences(train["ewe"])
        trainX = pad_sequences(trainX, maxlen=en_seq_length, padding='post')
        trainX = convert_to_tensor(trainX, dtype=int64)

        # Decoder
        dec_tokenizer = self.create_tokenizer(train["fr"])
        dec_seq_length = self.find_seq_length(train["fr"])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer)

        trainY = dec_tokenizer.texts_to_sequences(train["fr"])
        trainY = pad_sequences(trainY, maxlen=dec_seq_length, padding='post')
        trainY = convert_to_tensor(trainY, dtype=int64)

        return trainX, trainY, train, en_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size

data = PrepareDataset()
trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data("cleaned_dataset_ewe_fr_izzy.csv")

print(train_orig[0]["ewe"], "\n", trainX[0, :])