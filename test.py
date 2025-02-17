from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

messages = [
    # {
    #     "role": "system",
    #     "content": "You are a friendly chatbot who always responds in the style of a pirate",
    # },
    {"role": "user", "content": "Who are you"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
# outputs = model.generate(tokenized_chat, max_new_tokens=1024) 
# print(tokenizer.decode(outputs[0]))

