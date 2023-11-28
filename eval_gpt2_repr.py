from matplotlib import pyplot as plt
import numpy as np
import torch, random
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def preprocess():
    with open("./experiments/wh_adjunct/input.txt", 'r') as sentence_file:
        sentences = sentence_file.read().splitlines()
        return sentences[:120]
    
def train_test_sets(hiddens):
    that = hiddens[:20]
    random.shuffle(that)
    that = [("comp", v) for v in that]
    whether = hiddens[20:40]
    random.shuffle(whether)
    whether = [("comp", v) for v in whether]
    why = hiddens[40:60]
    random.shuffle(why)
    why = [("wh-adj", v) for v in why]
    how = hiddens[60:80]
    random.shuffle(how)
    how = [("wh-adj", v) for v in how]
    when = hiddens[80:100]
    random.shuffle(when)
    when = [("wh-adj", v) for v in when]
    where = hiddens[100:120]
    random.shuffle(where)
    where = [("wh-adj", v) for v in where]

    train = that[:15] + why[:15] + how[:15] + when[:15] + where[:15] + whether[:15]
    random.shuffle(train)
    test = that[15:] + why[15:] + how[15:] + when[15:] + where[15:] + whether[15:]
    random.shuffle(test)
    entire = that + whether + why + how + when + where
    random.shuffle(entire)

    return train, test, entire, that, whether, why, how, when, where
    
def logistic_regression(train, test):
    # Prepare the training data
    X_train = torch.stack([h for _, h in train])
    y_train = [label for label, _ in train]

    # Prepare the testing data
    X_test = torch.stack([h for _, h in test])
    y_test = [label for label, _ in test]

    # Convert to NumPy arrays if using scikit-learn
    X_train_np = X_train.detach().numpy()
    y_train_np = np.array(y_train)
    X_test_np = X_test.detach().numpy()
    y_test_np = np.array(y_test)

    # Initialize and train the logistic regression classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_np, y_train_np)

    # Predict on the test set and evaluate
    y_pred = clf.predict(X_test_np)
    print("Logistic Regression Classifier")
    print(classification_report(y_test_np, y_pred))
    return y_pred, y_test_np

def incr_logistic_regression(that, whether, why, how, when, where):

    all_y_test = []
    all_y_pred = []
    train = []
    while len(train) < 120:
        test = that[:5] + why[:5] + how[:5] + when[:5] + where[:5] + whether[:5]
        that = that[5:]
        whether = whether[5:]
        why = why[5:]
        how = how[5:]
        when = when[5:]
        where = where[5:]
        curr_train = train + that[5:] + why[5:] + how[5:] + when[5:] + where[5:] + whether[5:]

        y_pred, y_test_np = logistic_regression(train=curr_train, test=test)

        all_y_test.extend(y_test_np)
        all_y_pred.extend(y_pred)

        train = train + test

    print("Final Incremental Testing Report")
    print(classification_report(all_y_test, all_y_pred))
    

def incr_rlogistic_regression(whole):

    all_y_test = []
    all_y_pred = []
    index = 0
    while index < 120:
        test = whole[index:index+20]
        curr_train = whole[0:index] + whole[index+20:]
        random.shuffle(curr_train)

        y_pred, y_test_np = logistic_regression(train=curr_train, test=test)

        all_y_test.extend(y_test_np)
        all_y_pred.extend(y_pred)

        index += 20

    print("Final Incremental Testing Report")
    print(classification_report(all_y_test, all_y_pred))
    # conf_mat = confusion_matrix(all_y_test, all_y_pred, labels=["comp", "wh-adj"])
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["comp", "wh-adj"])
    # disp.plot(cmap=plt.cm.Blues)
    # plt.show()


def svm(train, test):
    # Prepare the training data
    X_train = torch.stack([h for _, h in train])
    y_train = [label for label, _ in train]

    # Prepare the testing data
    X_test = torch.stack([h for _, h in test])
    y_test = [label for label, _ in test]

    # Convert to NumPy arrays if using scikit-learn
    X_train_np = X_train.detach().numpy()
    y_train_np = np.array(y_train)
    X_test_np = X_test.detach().numpy()
    y_test_np = np.array(y_test)

    # Using a linear kernel first
    svm_clf = SVC(kernel='linear')
    svm_clf.fit(X_train_np, y_train_np)
    y_pred_svm = svm_clf.predict(X_test_np)
    print("Support Vector Machine Classifer")
    print(classification_report(y_test_np, y_pred_svm))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
model.eval()

sentences = preprocess()

# Each index denotes the hidden state of 'that', 'whether', 'where', 'when', 'why', or 'how' of sentence i + 1
hidden_states = []

for sentence in sentences:
    inputs_tokens = tokenizer(sentence, return_tensors="pt")
    output_tokens = model(**inputs_tokens)
    curr_hidden_states = output_tokens.last_hidden_state

    for word in ['that', 'where', 'when', 'how', 'why', 'whether']:
        word_ids = [i for i, token in enumerate(inputs_tokens['input_ids'][0]) if tokenizer.decode([token]).strip() == word]
        for word_id in word_ids:
            # Extract and process the hidden state for each word
            word_hidden_state = curr_hidden_states[0, word_id, :]
            hidden_states.append(word_hidden_state)

print(len(hidden_states))
# These tags map to the indices of the list of hidden_states
tags = ['comp'] * 40 + ['wh-adj'] * 80

train_set, test_set, entire, that_set, whether_set, why_set, how_set, when_set, where_set = train_test_sets(hiddens=hidden_states)

logistic_regression(train=train_set, test=test_set)
svm(train=train_set, test=test_set)
# incr_logistic_regression(that_set, whether_set, why_set, how_set, when_set, where_set)
incr_rlogistic_regression(whole=entire)