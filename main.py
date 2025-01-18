import re
from loader import load_text
from preprocessing import SimpleTokenizerV2  

def main():
    raw_text = load_text('the-verdict.txt')

    # preprocess text and create vocabulary
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer,token in enumerate(all_tokens)}

    tokenizer = SimpleTokenizerV2(vocab)

    # text1 = "Hello, do you like tea?"
    # text2 = "In the sunlit terraces of the palace."
    # text = " <|endoftext|> ".join([text1, text2])
    # print(text)

    # print(tokenizer.encode(text))
    # print(tokenizer.decode(tokenizer.encode(text)))

    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

if __name__ == '__main__':
    main()