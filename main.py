import torch

from loader import load_text
from dataloader import create_dataloader

def main():
    raw_text = load_text('the-verdict.txt')

    # START OF INPUT EMBEDDINGS PIPELINE
    
    max_len = 4 # the number of token ids in each of the input-target tensors; change this to 256 for the full text

    # Batch size controls the number of input-target pairs in each batch; this is a hyperparameter that can be tuned
    # During training, rather than processing the entire dataset at once, we break it into smaller batches of examples that are processed together.
    # This allows for more efficient use of memory and allows for more frequent updates to the model's parameters.
    batch_size = 8

    # Create the input-target pairs for training the llm
    dataloader = create_dataloader(
        text=raw_text, 
        batch_size=batch_size,
        max_len=max_len, 
        stride=max_len, # stride controls the overlap between input-target pairs, here it's set to max_len to ensure no overlap between input-target pairs
        shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs for the first batch:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    print("\nTargets shape:\n", targets.shape)

    # Create the token embeddings matri
    vocab_size = 50257 # gpt-2 vocab size
    output_dim = 256 # output dimension of each token

    # The token embeddings are a matrix of shape (batch_size, max_len, output_dim)
    # Each row in the matrix corresponds to a token in the input sequence
    # The output dimension is the dimension of the embedding space, which is 256 for the full text
    # The embedding layer is a learnable matrix, initialized with random weights, that maps each token to a vector in the embedding space
    # The embedding layer is essentially a lookup table that maps each token id to a vector in the embedding space
    # E.g., the first row of the embedding layer corresponds to the token id 0, the second row corresponds to the token id 1, etc.
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(f"Token embedding layer shape: {token_embedding_layer.weight.shape}")
    print(f"Token embedding layer rows: {token_embedding_layer.weight.shape[0]}")
    print(f"Token embedding layer columns: {token_embedding_layer.weight.shape[1]}")
    print(f"Token embedding layer weight:\n{token_embedding_layer.weight}")

    # Apply the token embedding layer to the inputs
    token_embeddings = token_embedding_layer(inputs)
    print("\nToken embeddings shape:\n", token_embeddings.shape)

    # Example: get the embeddings for the 1st, 2nd, and 4th tokens in the first batch
    sample = torch.tensor([1, 2, 4]) # should return 3 embedding vectors, each of shape (output_dim,)
    print(f"Token embeddings for the 1st, 2nd, and 4th tokens in the first batch: {token_embedding_layer(sample)}")

    # Create positional embeddings
    """When preparing a dataset for LLM training, the choice between these approaches affects both how you structure 
    your data and what the model can learn. If you use absolute embeddings, you might need to ensure your training 
    sequences don't exceed your maximum position encoding length. With relative embeddings, you have more flexibility 
    with sequence lengths, but you need to carefully consider how to handle long-range dependencies in your data."""
    context_length = max_len # in practice, context length is the supported maximum length of the input sequence for the llm
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(f"Positional embeddings shape: {pos_embeddings.shape}")

    # Add token embeddings and positional embeddings
    """But here's the key insight: in language, meaning comes from both what words mean AND where they appear. 
    By adding these embeddings together, we create a single representation that encodes both pieces of information 
    simultaneously.
    
    The input embeddings will then be used as input to the transformer model.
    """
    input_embeddings = token_embeddings + pos_embeddings
    print(f"Input embeddings shape: {input_embeddings.shape}")

    # END OF INPUT EMBEDDINGS PIPELINE

if __name__ == '__main__':
    main()