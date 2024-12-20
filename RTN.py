from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
import torch
import gc
from utils import evaluate, model_size

# token : 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

model = "meta-llama/Llama-2-7b-chat-hf"
# model = "meta-llama/Meta-Llama-3.1-8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True
)
q_model_nf = AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb_config, device_map="auto", use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=True)
print(q_model_nf)
gc.collect()
torch.cuda.empty_cache()

model_size(q_model_nf)  # 3588


q_model_nf_perplexity = evaluate(q_model_nf, tokenizer)
print(f"\nNF4 model perplexity: {q_model_nf_perplexity:.2f}")  # 7.86
gc.collect()
torch.cuda.empty_cache()