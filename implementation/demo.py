from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Define the checkpoint and tokenizer
checkpoint = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Define a maximum sequence length (e.g., 128 tokens)
max_length = 128

# Define a function to generate text
def generate_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=len(input_ids[0]) + max_new_tokens, num_return_sequences=1)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output
    

# Example usage
input_text = "Translate this English text to French: 'Hello, how are you?'"
output_text = generate_text(input_text)
print(output_text)
