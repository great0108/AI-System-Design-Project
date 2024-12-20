from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
import torch
import gc
from utils import evaluate, model_size

# token : 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

model = "meta-llama/Llama-2-7b-chat-hf"
# model = "meta-llama/Meta-Llama-3.1-8B"
q_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", use_auth_token=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=True)
print(q_model)
gc.collect()
torch.cuda.empty_cache()

model_size(q_model) # 12852

q_model_perplexity = evaluate(q_model, tokenizer)
print(f"\nFP16 model perplexity: {q_model_perplexity:.2f}") # 7.37
gc.collect()
torch.cuda.empty_cache()