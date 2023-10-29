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
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    with torch.no_grad():
        output = model.generate(**inputs)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

# Example usage
input_text = "Translate this English text to French: 'Hello, how are you?'"
output_text = generate_text(input_text)
print(output_text)
