from numpy import sort
from transformers import BertTokenizer, BertForMaskedLM, pipeline
import nltk
import matplotlib.pyplot as plt
import sys
import csv

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm = pipeline('fill-mask', model=model, tokenizer=tokenizer)


def read_sentences(path):
    with open(f"{path}/input.txt", 'r') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]

# nltk.download()

def mask_sentence(sentence: str):
    distribution = {}
    masked_data = mlm(sentence, top_k=15)
    for prediction in masked_data:
        
        prob = prediction['score']
        token = prediction['token_str']
        sentence = prediction['sequence']
        
        if token not in [',', '.', ';', ':', '!', '?', '``', '\"\"', '\"']:

            # MAY WANT TO USE ANOTHER TAGGER THAT DISTINGUISHES BETWEEN ARG AND ADJ

            nltk_tokenize = nltk.word_tokenize(sentence)
            tag = [tag for word, tag in nltk.pos_tag(nltk_tokenize) if word == token][0]

            print(f"{token}: {prob} {tag}")

            if tag not in [',', '.', ';', ':', '!', '?', '``', '\"\"', '\"']:
                if tag in distribution:
                    distribution[tag] += prob
                else:
                    distribution[tag] = prob
    
    return distribution

    # pos_tags = list(distribution.keys())
    # probabilities = list(distribution.values())

    # plt.bar(pos_tags, probabilities)

    # plt.title('Tag Probabilities')
    # plt.xlabel('Tags')
    # plt.ylabel('Tag Probability of Fillers')

    # plt.show()


def mask_sentence_token(sentence: str):
    distribution = []
    masked_data = mlm(sentence, top_k=15)
    for prediction in masked_data:
        
        prob = prediction['score']
        token = prediction['token_str']
        sentence = prediction['sequence']

        if token not in [',', '.', ';', ':', '!', '?', '``', '\"\"', '\"', '-', '...']:

            nltk_tokenize = nltk.word_tokenize(sentence)
            tag = [tag for word, tag in nltk.pos_tag(nltk_tokenize) if word == token][0]

            distribution.append((token, tag, prob))
    
    return distribution


def controller_pos(path: str):
    sentences = read_sentences(path)

    with open(f"{path}/output_pos.tsv", 'w', newline='', encoding='utf-8') as tsvfile:
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

    with open(f"{path}/output_word.tsv", 'w', newline='', encoding='utf-8') as tsvfile:
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
