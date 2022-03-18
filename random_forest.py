from sklearn.ensemble import RandomForestClassifier
from preprocessing import get_data
from sklearn.tree import DecisionTreeClassifier
import os
import glob


def clean_folder(path):
    files = glob.glob(path, recursive=True)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Ошибка: %s : %s" % (f, e.strerror))


def trees_predict(clf_f, x, y_proba_i, filename):
    f = open(filename, 'w')
    f.write(str(y_proba_i) + '\n')
    for estimator in clf_f.estimators_:
        f.write(str(estimator.predict_proba([x])[0][0])+'\n')


def write_to_files(clf_f, x, y_proba_i, k, correct=True):
    if correct:
        path = "input_data\\correct_x\\"
    else:
        path = "input_data\\incorrect_x\\"
    filename = path + str(k) + '.txt'
    trees_predict(clf_f, x, y_proba_i, filename)


clean_folder('input_data/correct_x/*.txt')
clean_folder('input_data/incorrect_x/*.txt')
x_train, y_train = get_data("data\\df_train.csv")
x_test, y_test = get_data("data\\df_test.csv")
clf = RandomForestClassifier(n_estimators=100, max_depth=5)
clf.fit(x_train, y_train)
y = clf.predict(x_test)
y_proba = clf.predict_proba(x_test)
k_true = 0
k_false = 0
for i in range(len(y)):
    if y[i] == y_test[i]:
        k_true += 1
        write_to_files(clf, x_test.loc[i], y_proba[i][0], k_true)
        s = "correct " + str(k_true)
    else:
        k_false += 1
        write_to_files(clf, x_test.loc[i], y_proba[i][0], k_false, False)
        s = "incorrect " + str(k_false)
print("correct prediction ", k_true)
print("incorrect prediction ", k_false)
