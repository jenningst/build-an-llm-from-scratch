from loader import load_text
from preprocessing import TokenizerV1  

def main():
    text = load_text('the-verdict.txt')
    tokenizer = TokenizerV1(text)

    text = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode("Hello, do you like tea?")
    print(ids)
    print(tokenizer.decode(ids))

if __name__ == '__main__':
    main()