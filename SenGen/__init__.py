import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
num_sentences = 100
def generate_captions(input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, temperature=0.8,
                             num_return_sequences=num_sentences, do_sample=True, max_new_tokens=128, top_k=10)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


source_concept = "cat"
source_text = f"Provide a caption for images containing a {source_concept}. "
"The captions should be in English and should be no longer than 150 characters."

source_captions = generate_captions(source_text)

for source_caption in source_captions :