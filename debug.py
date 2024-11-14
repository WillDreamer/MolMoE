from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, GenerationConfig
import torch
from data_pipeline import conversation_lib
from transformers import TrainingArguments


tokenizer = AutoTokenizer.from_pretrained("downloads/Llama-3.2-1B-Instruct")
model = LlamaForCausalLM.from_pretrained("downloads/Llama-3.2-1B-Instruct")
generation = GenerationConfig.from_pretrained("downloads/Llama-3.2-1B-Instruct")
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Who are you"},
]

print(model)
print(model.config)

prompt = tokenizer.apply_chat_template(messages, tokenize=False)

print(prompt)

# input_ids = tokenizer.encode(prompt, return_tensors="pt")
# print(input_ids)
# output = model.generate(
#     input_ids, 
#     generation_config=generation,
#     max_new_tokens=200
#     )


# print(output)
# print(tokenizer.decode(output[0]))

'''
<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 06 Oct 2024

You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Who are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'm an AI assistant, and I'm here to provide information and answer your questions to the best of my knowledge. 
I'm a large language model, I don't have a personal identity or a physical presence, but I'm here to help and assist 
with any questions or topics you'd like to discuss. How can I assist you today?<|eot_id|>
'''

conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
conv = conversation_lib.default_conversation.copy()
conv.append_message(conv.roles[0], messages[1]["content"])
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()

print(prompt)

