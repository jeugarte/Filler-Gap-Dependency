import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def extract_hidden_states(hidden_file, indices):
    hidden_states = []
    with open(hidden_file, 'r') as file:
        lines = file.readlines()
        for index in indices:
            hidden_states.append([float(num) for num in lines[index].split()])
    return np.array(hidden_states)

def train_and_evaluate(X_train, y_train, X_test, y_test):
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_test, y_pred

def linear_regression(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Logistic Regression Classifier")
    print(classification_report(y_test, y_pred))
    return y_test,y_pred

def main(hidden_file):

    # list of indices denoting the comp/filler index in experiment 1 sentences
    indices = [4, 23, 42, 60, 79, 97, 116, 134, 152, 171, 190, 208, 227, 246, 265, 283, 302, 
               319, 338, 356,
                1105, 1124, 1143, 1161, 1180, 1198, 1217, 1235, 1253, 1272, 1291, 1309, 1328, 
                1347, 1366, 1384, 1403, 1420, 1439, 1457,
                  1847, 1866, 1885, 1903, 1922, 1940, 1959, 1977, 1995, 2014, 2033, 2051, 2070, 2089, 2108, 
                  2126, 2145, 2162, 2181, 2199,]
               
                #that
            #   4, 23, 42, 60, 79, 97, 116, 134, 152, 171, 190, 208, 227, 246, 265, 283, 302, #17 
            #   319, 338, 356,

               #whether
            #    375, 393, 411, 429, 447, 465, 482, 499, 517, 536, 554, 572, 590, #33
            #    609, 627, 645, 664, 681, 698, 716,

                #why
            #   734, 753, 772, 790, 809, 827, 846, 864, 882, 
            #   901, 920, 938, 957, 976, 995, 1013, 1032, 1049, 1068, 1086,

                #how 
            #    1105, 1124, 1143, 1161, 1180, 1198, 1217, 1235, 1253, 1272, 1291, 1309, 1328, 1347, 1366, 1384, 
            #    1403, 1420, 1439, 1457

            #   when
            #   1476, 1495, 1514, 1532, 1551, 1569, 1588, 1606, 1624, 
            #     1643, 1662, 1680, 1699, 1718, 1737, 1755, 1774, 1791, 1810, 1828,

            #where
            #   1847, 1866, 1885, 1903, 1922, 1940, 1959, 1977, 1995, 2014, 2033, 2051, 2070, 2089, 2108, 
            #  2126, 2145, 2162, 2181, 2199,
                  
                  
                #    1105, 1124, 1143, 
                #  1161, 1180, 1198, 1217, 1235, 1253, 1272, 1291, 1309, 1328, 1347, 1366, 1384, 
                #  1403, 1420, 1439, 1457, 1476, 1495, 1514, 1532, 1551, 1569, 1588, 1606, 1624, 
                #  1643, 1662, 1680, 1699, 1718, 1737, 1755, 1774, 1791, 1810, 1828, 1847, 1866, 
                #  1885, 1903, 1922, 1940, 1959, 1977, 1995, 2014, 2033, 2051, 2070, 2089, 2108, 
                #  2126, 2145, 2162, 2181, 2199, 2218, 2234, 2250, 2266, 2282, 2298, 2314, 2330, 
                #  2346, 2362, 2378, 2394, 2410, 2426, 2442, 2458, 2474, 2489, 2505, 2521, 2537, 
                #  2553, 2569, 2584, 2600, 2615, 2631, 2646, 2661, 2677, 2693, 2708, 2724, 2740, 
                #  2756, 2771, 2787, 2802, 2818, 2833, 2849, 2865, 2881, 2897, 2913, 2929, 2945, 
                #  2961, 2977, 2993, 3009, 3025, 3041, 3057, 3073, 3089, 3105, 3120, 3136, 3152, 
                #  3168, 3184, 3200, 3215, 3231, 3246, 3262, 3277, 3292, 3308, 3324, 3339, 3355, 
                #  3371, 3387, 3402, 3418, 3433, 3449, 3464]

    X = extract_hidden_states(hidden_file, indices)

    # labels
    y1 = ['COMP'] * 20 + ['ADJ'] * 40

    class_indices_1 = {}
    for i, condition in enumerate(y1):
        if condition not in class_indices_1:
            class_indices_1[condition] = []
        class_indices_1[condition].append(i)

    all_y_true = []
    all_y_pred = []

    reserved_test_indices = [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
        # 0,1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19]
        
        # 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]

        #40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59


    total_indices = set(range(len(y1)))
    remaining_indices = total_indices - set(reserved_test_indices)

    while remaining_indices:
        additional_test_indices = np.random.choice(list(remaining_indices), min(5, len(remaining_indices)), replace=False)

        test_indices = list(reserved_test_indices) + list(additional_test_indices)
        remaining_indices -= set(additional_test_indices)

        X_test = X[test_indices]
        y1_test = np.array(y1)[test_indices]

        train_indices = [idx for idx in total_indices if idx not in test_indices]
        X_train = X[train_indices]
        y1_train = np.array(y1)[train_indices]

        print("Results for current round:")
        y_true_round, y_pred_round = train_and_evaluate(X_train, y1_train, X_test, y1_test)
        all_y_true.extend(y_true_round)
        all_y_pred.extend(y_pred_round)

    conf_mat = confusion_matrix(all_y_true, all_y_pred, labels=["COMP", "ADJ"])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["COMP", "ADJ"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

if __name__ == "__main__":
    hidden_file = sys.argv[1]
    main(hidden_file)