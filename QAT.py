import torch
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer, Int4WeightOnlyQATQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
import torch
import gc
from utils import evaluate, model_size
from datasets import load_dataset

# token : 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

model = "meta-llama/Llama-3.1-8B"
q_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", use_auth_token=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=True)
print(q_model)

trainenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
trainenc = tokenizer("\n\n".join(trainenc['text']), return_tensors='pt')

trainenc = trainenc.input_ids.to(q_model.device)
print(trainenc[0])


# Quantizer for int8 dynamic per token activations +
# int4 grouped per channel weights, only for linear layers
qat_quantizer = Int4WeightOnlyQATQuantizer()

# Insert "fake quantize" operations into linear layers.
# These operations simulate quantization numerics during
# training without performing any dtype casting
model = qat_quantizer.prepare(q_model)

# # Standard training loop
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
# loss_fn = torch.nn.CrossEntropyLoss()
# for i in range(10):
#     example = torch.randint(0, 4096, (2, 16)).cuda()
#     target = torch.randn((2, 16, 4096)).cuda()
#     output = model(example)
#     loss = loss_fn(output, target)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()

# Convert fake quantize to actual quantize operations
# The quantized model has the exact same structure as the
# quantized model produced in the corresponding PTQ flow
# through `Int8DynActInt4WeightQuantizer`
model = qat_quantizer.convert(model)
model_size(model)

model_perplexity = evaluate(model, tokenizer)
print(f"\nQAT model perplexity: {model_perplexity:.2f}")
gc.collect()
torch.cuda.empty_cache()