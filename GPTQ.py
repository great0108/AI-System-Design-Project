from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
from utils import evaluate, model_size

model = "meta-llama/Llama-2-7b-chat-hf"
# model = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
q_model_gptq = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quantization_config)

for para in q_model_gptq.parameters():
    para.requires_grad = False
q_model_gptq.config.use_cache = False
q_model_gptq.eval()
#print(q_model_gptq)

# q_model_gptq.save_pretrained("opt-125m-gptq")
# tokenizer.save_pretrained("opt-125m-gptq")

# q_model_gptq = AutoModelForCausalLM.from_pretrained("opt-125m-gptq", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("opt-125m-gptq", use_fast=False)
print(q_model_gptq)
model_size(q_model_gptq) # 2935


q_model_gptq_perplexity = evaluate(q_model_gptq, tokenizer)
print(f"\n GPTQ 4-bit model perplexity: {q_model_gptq_perplexity:.2f}") # 9.10