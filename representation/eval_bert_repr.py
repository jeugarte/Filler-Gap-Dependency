import sys
from matplotlib import pyplot as plt
import numpy as np
import torch, random
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def preprocess(hidden_file):
    with open(hidden_file, 'r') as sentence_file:
        sentences = sentence_file.read().splitlines()
        return sentences

def extract_hidden(sentences, tokenizer, model):
    hidden_states = []

    for sentence in sentences:
        inputs_tokens = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            output_tokens = model(**inputs_tokens)
        curr_hidden_states = output_tokens.last_hidden_state

        for word_index in range(curr_hidden_states.size(1)):
            input_word = tokenizer.decode([inputs_tokens['input_ids'][0, word_index]]).strip()
            if input_word.strip() in ['that', 'where', 'when', 'how', 'why', 'whether']:
                word_hidden_state = curr_hidden_states[0, word_index, :].tolist()
                hidden_states.append(word_hidden_state)

    return hidden_states

def extract_sets(key, hidden):

    states = []

    if key == "that":
        states = hidden[:20]
        random.shuffle(states)
        states = [("COMP", v) for v in states]

    elif key == "whether":
        states = hidden[20:40]
        random.shuffle(states)
        states = [("COMP", v) for v in states]

    elif key == "why":
        states = hidden[40:60]
        random.shuffle(states)
        states = [("ADJ", v) for v in states]

    elif key == "how":
        states = hidden[60:80]
        random.shuffle(states)
        states = [("ADJ", v) for v in states]

    elif key == "when":
        states = hidden[80:100]
        random.shuffle(states)
        states = [("ADJ", v) for v in states]

    elif key == "where":
        states = hidden[100:120]
        random.shuffle(states)
        states = [("ADJ", v) for v in states]

    return states


def train_test_sets(hiddens):
    that = hiddens[:20] + hiddens[160:200]
    random.shuffle(that)
    that = [("COMP", v) for v in that]
    whether = hiddens[20:40]
    random.shuffle(whether)
    whether = [("COMP", v) for v in whether]
    why = hiddens[40:60]
    random.shuffle(why)
    why = [("ADJ", v) for v in why]
    how = hiddens[60:80]
    random.shuffle(how)
    how = [("ADJ", v) for v in how]
    when = hiddens[80:100] + hiddens[120:140]
    random.shuffle(when)
    when = [("ADJ", v) for v in when]
    where = hiddens[100:120] + hiddens[140:160]
    random.shuffle(where)
    where = [("ADJ", v) for v in where]

    train = that[:15] + why[:15] + how[:15] + when[:15] + where[:15]
    random.shuffle(train)
    test = that[15:] + why[15:] + how[15:] + when[15:] + where[15:]
    random.shuffle(test)
    entire = that + whether + why + how + when + where
    random.shuffle(entire)
    print(len(entire))

    return train, test, entire, that, whether, why, how, when, where
    
def logistic_regression(train, test):
    X_train = np.array([h for _, h in train])
    y_train = [label for label, _ in train]

    X_test = np.array([h for _, h in test])
    y_test = [label for label, _ in test]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print("Logistic Regression Classifier")
    # print(classification_report(y_test, y_pred))
    return y_pred, y_test

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
    while index < 200:
        test = whole[index:index+40]
        curr_train = whole[0:index] + whole[index+40:]
        random.shuffle(curr_train)

        y_pred, y_test_np = logistic_regression(train=curr_train, test=test)

        all_y_test.extend(y_test_np)
        all_y_pred.extend(y_pred)

        index += 40

    print("Final Incremental Testing Report")
    print(classification_report(all_y_test, all_y_pred))


def svm(train, test):
    X_train = np.array([h for _, h in train])
    y_train = [label for label, _ in train]

    X_test = np.array([h for _, h in test])
    y_test = [label for label, _ in test]

    svm_clf = SVC(kernel='linear')
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    # print("Support Vector Machine Classifer")
    # print(classification_report(y_test, y_pred_svm))
    return y_pred_svm, y_test



def controller_singular(hidden_file, train_comp_name, train_wh_name, test_name):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    sentences = preprocess(hidden_file=hidden_file)
    hidden_states = extract_hidden(sentences=sentences, tokenizer=tokenizer, model=model)

    train_comp = extract_sets(key=train_comp_name, hidden=hidden_states)
    train_wh = extract_sets(key=train_wh_name, hidden=hidden_states)
    train = train_comp + train_wh
    test = extract_sets(key=test_name, hidden=hidden_states)

    all_y_true = []
    all_lr_pred = []
    all_svm_pred = []
    tested = train

    while tested:
        additional_test_i = np.random.choice(len(tested), min(5, len(tested)), replace=False)
        additional_test = [tested[i] for i in additional_test_i]
        tested = [tested[i] for i in range(len(tested)) if i not in additional_test_i]

        test_round = test + list(additional_test)
        train_round = [state for state in train if state not in additional_test]

        pred_lr, true = logistic_regression(train=train_round, test=test_round)
        pred_svm, _ = svm(train=train_round, test=test_round)

        all_y_true.extend(true)
        all_lr_pred.extend(pred_lr)
        all_svm_pred.extend(pred_svm)

    # print("\nResults:\n")
    # print("Support Vector Machine Classifer")
    # print(classification_report(all_y_true, all_svm_pred))
    # print("\n\n")
    # print("Logistic Regression Classifier")
    # print(classification_report(all_y_true, all_lr_pred))

    # conf_mat = confusion_matrix(all_y_true, all_lr_pred, labels=["COMP", "ADJ"])
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["COMP", "ADJ"])
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title(f"{train_comp_name} & {train_wh_name} predict {test_name}")
    # plt.show()

    # conf_mat = confusion_matrix(all_y_true, all_svm_pred, labels=["COMP", "ADJ"])
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["COMP", "ADJ"])
    # disp.plot(cmap=plt.cm.Reds)
    # plt.title(f"{train_comp_name} & {train_wh_name} predict {test_name}")
    # plt.show()

    return all_y_true, all_lr_pred, all_svm_pred



def controller_full(hidden_file, test):
    base_wh = ['why', 'how', 'when', 'where']
    base_comp = ['that', 'whether']

    whs = [item for item in base_wh if item != test]
    comps = [item for item in base_comp if item != test]

    all_y_true = []
    all_lr_pred = []
    all_svm_pred = []

    for comp in comps:
        for wh in whs:
            curr_true, curr_lr, curr_svm = controller_singular(hidden_file, train_comp_name=comp, train_wh_name=wh, test_name=test)
            all_y_true.extend(curr_true)
            all_lr_pred.extend(curr_lr)
            all_svm_pred.extend(curr_svm)

    print("\nResults:\n")
    print("Logistic Regression Classifier")
    print(classification_report(all_y_true, all_lr_pred))
    print("\n\n")
    
    conf_mat = confusion_matrix(all_y_true, all_lr_pred, labels=["COMP", "ADJ"])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["COMP", "ADJ"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Predict {test} w/ LRC")
    plt.show()

    print("Support Vector Machine Classifer")
    print(classification_report(all_y_true, all_svm_pred))

    
    conf_mat = confusion_matrix(all_y_true, all_svm_pred, labels=["COMP", "ADJ"])

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["COMP", "ADJ"])
    plt.rcParams.update({'font.size': 32})

    disp.plot(cmap=plt.cm.Reds, values_format='.0f')
    plt.title(f"Predict {test} w/ SVC")
    plt.show()


if __name__ == "__main__":
    hidden_file = sys.argv[1]
    singular_or_full = sys.argv[2]

    if singular_or_full == "full":
        if len(sys.argv) < 4:
            raise Exception("Not enough arguments provided for 'full' mode")
        test_name = sys.argv[3]
        
        if test_name not in ["when", "where", "how", "why", "that", "whether"]:
            raise Exception("Choose either \"that\", \"whether\", \"when\", \"where\", \"how\", or \"why\" as the complementizer")
        controller_full(hidden_file, test=test_name)

    else:
        if len(sys.argv) < 6:
            raise Exception("Not enough arguments provided for 'singular' mode")
        train_comp_name = sys.argv[3]
        train_wh_name = sys.argv[4]
        test = sys.argv[5]

        if train_comp_name not in ["that", "whether"]:
            raise Exception("Choose either \"that\" or \"whether\" as the complementizer")
       
        if train_wh_name not in ["when", "where", "how", "why"]:
            raise Exception("Choose either \"when\", \"where\", \"how\", or \"why\" as the wh-adjunct")
        
        if (test not in ["when", "where", "how", "why", "that", "whether"]) or (test in ["when", "where", "how", "why", "that", "whether"] and (test == train_wh_name or test == train_comp_name)):
            raise Exception("Choose a unique test")

        controller_singular(hidden_file, train_comp_name, train_wh_name, test_name=test)