"""
Created by Analitika at 09/08/2024
contact@analitika.fr
"""

# External imports
import requests
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Internal imports
from config import HUGGINGFACE_TOKEN

# check whether a GPU is available
is_gpu = torch.cuda.is_available()
if is_gpu:
    print("GPU is available")


# Load tokenizer and model
model_str = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_str)
model = AutoModelForMaskedLM.from_pretrained(model_str)

# Prepare text
text = "Machine Learning es el mejor [MASK]"
encoded_input = tokenizer(text, return_tensors="pt")
len(encoded_input["input_ids"][0])

# See the tokens
tkns = [tokenizer.decode([token_id]).strip() for token_id in encoded_input["input_ids"][0]]

# Compute embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Get the predicted token ID
masked_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
# predicted_token_id = model_output.logits[0, masked_index].argmax(axis=-1)  # top one
logits = model_output.logits[0, masked_index, :]

# Get the top 5 token IDs with the highest probability
top_5_token_ids = logits.topk(5, dim=-1).indices[0].tolist()

# Convert the top 5 token IDs to corresponding words
top_5_tokens = [tokenizer.decode([token_id]).strip() for token_id in top_5_token_ids]

# Display the original text and the top 3 predictions
print(f"Texto Original: {text}")
print("Top 5 Predicciones:")
for i, token in enumerate(top_5_tokens, start=1):
    completed_text = text.replace(tokenizer.mask_token, token)
    print(f"{i}: {completed_text}")


# Call to the HuggingFace API
API_URL = f"https://api-inference.huggingface.co/models/{model_str}"


def query(payload, url):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()


data = query({"inputs": text}, API_URL)
for i, ans in enumerate(data):
    completed_text = ans["sequence"]
    print(f"{i}: {completed_text}")


