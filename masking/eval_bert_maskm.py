from transformers import BertModel, BertTokenizer, BertForMaskedLM, pipeline
import nltk
import torch
import matplotlib.pyplot as plt
import spacy


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
nlp = spacy.load("en_core_web_sm")

sentences = ["The intern stupidly forgot [MASK] the surgeon found."]

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors='pt')
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    token_logits = model(**inputs).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_tokens:
    print(tokenizer.decode([token]))

# nltk.download()
mlm = pipeline('fill-mask', model="bert-base-uncased")


distribution = {}
masked_data = mlm(sentences[0], top_k=15)
for prediction in masked_data:
    prob = prediction['score']
    token = prediction['token_str']
    sentence = prediction['sequence']
    
    tags = nlp(sentence)
    for token in tags:
        print(f"{token.text}: {token.pos_} ({token.head.text})")
    # tag = [tag for word, tag in nltk.pos_tag(nltk_tokenize) if word == token][0]

    # print(f"{token}: {prob} {tag}")

#     if tag not in [',', '.', ';', ':']:
#         if tag in distribution:
#             distribution[tag] += prob
#         else:
#             distribution[tag] = prob

# pos_tags = list(distribution.keys())
# probabilities = list(distribution.values())

# plt.bar(pos_tags, probabilities)

# plt.title('Tag Probabilities')
# plt.xlabel('Tags')
# plt.ylabel('Tag Probability of Fillers')

# plt.show()


