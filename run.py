#!/usr/bin/python3
from __future__ import print_function
import os
import sys
import glob
import functools

#import rfutils
import pandas as pd

UNK_TOKENS = {"<unk>", "<UNK>"}
FINAL_TOKENS = {"<eos>", "</S>", "</s>"}

# For running on Jenova
MODEL = {
    'gpt3': [
        "python ~/cs4745/pilot/Filler-Gap-Pilot/eval_gpt3.py --surprisalmode True --prefixfile {input_path} --outf {output_path}",
    ]
}
    
def do_system_calls(cmds):
    for cmd in cmds:
        # We're making calls for effect; redirect stdout to stdout
        print("Running command: %s" % cmd, file=sys.stderr)
        os.system(cmd)
        #print(rfutils.system_call(cmd))

def run_model(model_name, input_path, output_path):
    return do_system_calls(
        cmd.format(input_path=input_path, output_path=output_path)
        for cmd in MODEL[model_name]
    )

def sentences(words):
    def gen():
        sentence = []
        for word in words:
            sentence.append(word)
            if is_final(word):
                yield tuple(sentence)
                sentence.clear()
    return map(" ".join, gen())

def is_unk(w):
    return w in UNK_TOKENS

def is_final(w):
    return w in FINAL_TOKENS

def run_models(path, conditions_df, models):
    # Write sentences to a txt file to be fed to the LSTMs
    input_filename = os.path.join(path, "input_obj.txt")
    # with open(input_filename, 'wt') as outfile:
    #     for sentence in sentences(conditions_df['word']):
    #         print(sentence, file=outfile)
            
    # Run the LSTMs by command line invocation
    def output_dfs():
        for model in models:
            output_filename = os.path.join(path, "%s_output.tsv" % model)
            print(output_filename)
            run_model(model, input_filename, output_filename)
            output_df = pd.read_csv(
                os.path.join(output_filename),
                sep="\t",
                header=None,
                index_col=None,
                names=['model_word', 'surprisal']
            )
            output_df['model'] = model
            yield output_df

    # Combine results with conditions
    df = pd.concat(list(output_dfs()), ignore_index=True)

    return df


def main(path, *models):
    if not models:
        models = MODEL
    path = os.path.abspath(path)
    # Read in the data
    conditions_df = pd.read_csv(
        os.path.join(path, "items.tsv"),
        sep="\t",
    )
    df = run_models(path, conditions_df, models)
    df.to_csv(os.path.join(path, "combined_results2.csv"))

if __name__ == "__main__":
    main(*sys.argv[1:])