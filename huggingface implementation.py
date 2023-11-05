from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Load the pre-trained model and tokenizer
model_base_name = "meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(model_base_name)
tokenizer = LlamaTokenizer.from_pretrained(model_base_name)

# Check vocabulary size and maximum sequence length
vocab_size = tokenizer.vocab_size
max_seq_length = model.config.max_position_embeddings
print("Vocabulary Size:", vocab_size)
print("Max Sequence Length:", max_seq_length)

# Add a padding token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Specify input sentences
sentences = ["This is me", "A 2nd sentence"]

# Tokenize the input sentences with padding and truncation
input_ids = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length)['input_ids']

# Ensure token IDs are within the vocabulary range
input_ids = input_ids.clamp(max=vocab_size - 1)

# Get model outputs (logits)
with torch.no_grad():
    outputs = model(input_ids)

# Extract hidden states from the base model
hidden_states = outputs.logits

# Extract embeddings for [CLS] tokens (you can choose other tokens as needed)
cls_embeddings = hidden_states[:, 0, :]

# Now, cls_embeddings contains dense embeddings for your input sentences
# Compute cosine similarity using torch.nn.functional.cosine_similarity
import torch.nn.functional as F
similarity = F.cosine_similarity(cls_embeddings[0].unsqueeze(0), cls_embeddings[1].unsqueeze(0))
print("Cosine Similarity:", similarity.item())