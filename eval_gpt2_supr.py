import argparse
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast, AutoTokenizer, GPT2Config, GPT2Model
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Mask-based evaluation: extracts softmax vectors for specified words')

parser.add_argument('--seed', type=int, 
                    default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--temperature', type=float, 
                    default=1.0, help='temperature - higher will increase diversity')
parser.add_argument('--outf', type=str, 
                    default='generated.txt', help='output file for generated text')
parser.add_argument('--prefixfile', type=str, 
                    default='', help='File with sentence prefix from which to generate continuations')
parser.add_argument('--surprisalmode', type=bool, 
                    default=False, help='Run in surprisal mode; specify sentence with --prefixfile')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()


def compute_surprisal(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    surprisals = [-torch.log(probabilities[i, input_ids[0, i]]).item() for i in range(input_ids.size(1))]
    return surprisals

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with open(args.prefixfile, 'r') as f:
    raw_sentences = f.readlines()
sentences = [sentence.replace("<eos>", "<|endoftext|>").strip() for sentence in raw_sentences]

if args.surprisalmode:
    with open(args.outf, 'w') as outf:
        for sentence in sentences:
            torch.manual_seed(args.seed)
            tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)
            input_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long).unsqueeze(0).to(device)
            totalsurprisal = 0.0
            first_token_id = tokenized_sentence[0]
            input_ids = torch.cat((input_ids, torch.tensor([[first_token_id]], dtype=torch.long).to(device)), dim=1)
            outf.write(tokenizer.decode([first_token_id]) + "\t0.00\n")
            for token_id in tokenized_sentence[1:]:
                with torch.no_grad():
                    outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                word_surprisals = -1 * torch.log2(probs)
                word_surprisal = word_surprisals[0, token_id].item()

                token_text = tokenizer.decode([token_id])
                # Post-processing before writing to output file
                if token_text not in [" ", "ologist", "ably", "ator", "ly", "ener"]:
                    if token_text in [" stupid", " foolish"]:
                        outf.write(token_text + "ly" + "\t" + str(word_surprisal) + "\n")
                    elif token_text in [" regrett"]:
                        outf.write(token_text + "ably" + "\t" + str(word_surprisal) + "\n")
                    elif token_text in [" archae"]:
                        outf.write(token_text + "ologist" + "\t" + str(word_surprisal) + "\n")
                    elif token_text in [" excav"]:
                        outf.write(token_text + "ator" + "\t" + str(word_surprisal) + "\n")
                    elif token_text in [" gard"]:
                        outf.write(token_text + "ener" + "\t" + str(word_surprisal) + "\n")
                    else:
                        outf.write(token_text + "\t" + str(word_surprisal) + "\n")

                # outf.write(tokenizer.decode([token_id]) + "\t" + str(word_surprisal) + "\n")
                input_ids = torch.cat((input_ids, torch.tensor([[token_id]], dtype=torch.long).to(device)), dim=1)
