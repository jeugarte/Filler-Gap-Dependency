from transformers import BertModel, BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

sentences = ["The cleaner obviously knew [MASK] the resident fixed."]

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors='pt')
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    token_logits = model(**inputs).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_tokens:
    print(tokenizer.decode([token]))
