import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class TextProcessor:
    def __init__(self):
        self.tokenizer = lambda x: x.split()

    def TokenIterator(self, textData):
            for text in textData:
                yield self.tokenizer(text)

    def GenerateVocabulary(self, textData):
        vocabulary = build_vocab_from_iterator(self.TokenIterator(textData), min_freq = 2,specials=["<pad>","<unk>"])
        vocabulary.set_default_index(vocabulary["<unk>"])
        print("Vocabulary size:", len(vocabulary))
        #print("Vocabulary:", vocabulary.get_itos())
        return vocabulary

    def TextToTensor(self, text, vocabulary):
        tokens = self.tokenizer(text)
        return torch.tensor([vocabulary[token] for token in tokens], dtype=torch.long)

    def GeneratePaddings(self, textData, vocabulary):
        tokenizedTextData = [self.TextToTensor(text, vocabulary) for text in textData]
        textPadded = pad_sequence(tokenizedTextData, batch_first=True)
        print("Padded text shape:", textPadded.shape)
        #print("Padded text:", textPadded)
        return textPadded