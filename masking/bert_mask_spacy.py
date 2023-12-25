from numpy import sort
from transformers import BertTokenizer, BertForMaskedLM, pipeline
import spacy
import matplotlib.pyplot as plt
import sys
import csv
import string

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm = pipeline('fill-mask', model=model, tokenizer=tokenizer)
nlp = spacy.load("en_core_web_sm")

def read_sentences(path):
    with open(f"{path}/input.txt", 'r') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]

def mask_sentence(sentence: str):
    distribution = {}
    masked_data = mlm(sentence, top_k=15)
    word_predictions = [prediction for prediction in masked_data if prediction['token_str'] not in string.punctuation and prediction['token_str'] not in ['...']]
    total_prob = sum(prediction['score'] for prediction in word_predictions)

    for prediction in word_predictions:
        prob = prediction['score'] / total_prob
        token = prediction['token_str']
        sentence_with_prediction = prediction['sequence']

        tags = nlp(sentence_with_prediction)
        tag = [word.pos_ for word in tags if word.text == token][0]

        print(f"{token}: {prob} {tag}")

        if tag in distribution:
            distribution[tag] += prob
        else:
            distribution[tag] = prob

    return distribution


def mask_sentence_token(sentence: str):
    distribution = []
    masked_data = mlm(sentence, top_k=15)
    word_predictions = [prediction for prediction in masked_data if prediction['token_str'] not in string.punctuation and prediction['token_str'] not in ['...']]
    total_prob = sum(prediction['score'] for prediction in word_predictions)

    for prediction in word_predictions:
        prob = prediction['score'] / total_prob
        token = prediction['token_str']
        sentence_with_prediction = prediction['sequence']

        tags = nlp(sentence_with_prediction)
        tag = [word.pos_ for word in tags if word.text == token][0]

        distribution.append((token, tag, prob))
    
    return distribution


def controller_pos(path: str):
    sentences = read_sentences(path)

    with open(f"{path}/output_pos_spacy.tsv", 'w', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['sentence_id', 'pos_tag', 'probability'])

        sentence_id = 1

        for sentence in sentences:
            probabilities = mask_sentence(sentence)

            for pos in probabilities.keys():
                writer.writerow([sentence_id, pos, probabilities[pos]])
            
            sentence_id += 1


def controller_token(path: str):
    sentences = read_sentences(path)

    with open(f"{path}/output_word_spacy.tsv", 'w', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['sentence_id', 'token', 'pos_tag', 'probability'])

        sentence_id = 1

        for sentence in sentences:
            probabilities = mask_sentence_token(sentence)

            for token, pos, prob in probabilities:
                writer.writerow([sentence_id, token, pos, prob])
            
            sentence_id += 1



if __name__ == "__main__":
    token_or_pos = sys.argv[1]
    path = sys.argv[2]

    if token_or_pos == "pos":
        controller_pos(path)
    else:
        controller_token(path)
