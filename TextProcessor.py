import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class TextProcessor:
    def __init__(self):
        self.tokenizer = get_tokenizer('basic_english')

    def TokenIterator(self, textData):
            for text in textData:
                yield self.tokenizer(text)

    def GenerateVocabulary(self, textData):
        vocabulary = build_vocab_from_iterator(self.TokenIterator(self, textData), min_freq = 2,specials=["<pad>","<unk>"])
        vocabulary.set_default_index(vocabulary["<unk>"])
        print("Vocabulary size:", len(vocabulary))
        print("Vocabulary:", vocabulary.get_itos())
        return vocabulary

    def TextToTensor(text, vocabulary, tokenizer):
        tokens = tokenizer(text)
        return torch.tensor([vocabulary[token] for token in tokens], dtype=torch.long)

    def GeneratePaddings(self, textData):
        vocabulary = self.GenerateVocabulary(textData)
        tokenizedTextData = [self.TextToTensor(text, vocabulary, self.tokenizer) for text in textData]
        textPadded = pad_sequence(tokenizedTextData, batch_first=True)
        print("Padded text shape:", textPadded.shape)
        print("Padded text:", textPadded)
        return textPadded