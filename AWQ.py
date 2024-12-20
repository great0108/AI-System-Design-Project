from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from utils import evaluate, model_size

model_path = "meta-llama/Llama-3.1-8B"
#quant_path = 'opt-125m-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 3}

# Load model
q_model_awq = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

test_awq_model = q_model_awq
for para in test_awq_model.parameters():
    para.requires_grad = False

q_model_awq.quantize(tokenizer, quant_config=quant_config)

print(q_model_awq)
model_size(q_model_awq)


q_model_awq_perplexity = evaluate(q_model_awq, tokenizer)
print(f"\n AWQ 4-bit model perplexity: {q_model_awq_perplexity:.2f}") # 10.32