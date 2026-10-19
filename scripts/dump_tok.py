from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("HF1BitLLM/Llama3-8B-1.58-100B-tokens")
chat = [
    {"role": "system", "content": "You are a helpful, knowledgeable, and friendly AI assistant. You provide clear, concise, and accurate responses."},
    {"role": "user", "content": "Hello"}
]
prompt = t.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print("STRING PROMPT:")
print(repr(prompt))

tokens = t.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
print("TOKENS:")
print(tokens)
