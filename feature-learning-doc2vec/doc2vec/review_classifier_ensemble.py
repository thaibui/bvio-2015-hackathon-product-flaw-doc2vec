from gensim_infer_vector.gensim.models.doc2vec import Doc2Vec
from numpy.random import shuffle
from sklearn import svm, cross_validation, linear_model
from sklearn.externals import joblib
from mlxtend.sklearn import EnsembleClassifier
from sklearn.metrics import classification_report

class VectorizedData(object):
    data = []
    def __init__(self, filename, doc2vec):
        self.filename = filename
        for uid, line in enumerate(open(self.filename)):
            columns = line.split("\t")
            text = columns[2]
            words = []
            # sanitize the words
            for w in text.split(" "):
                r = w.lower()
                t = ''
                for c in r:
                    if c != "." and c != "," and c != "!":
                        t += c
                words.append(t)

            id = columns[0]
            if id in model.vocab:
                row = (columns[0], columns[1], doc2vec[id], columns[3], columns[4], columns[5])
                self.data.append(row)
        print "Vectorized Data loaded containing %d rows" % len(self.data)

    def shuffle(self):
        shuffle(self.data)

inTrainingData = "../data/review/dict_subsampled_400k-1.txt"
inModelName = "../model/20k-4_500_40_dbow_negative15_window16.doc2vec"

# Load the model
model = Doc2Vec.load(inModelName)

# Prepare training data
trainingData = VectorizedData(inTrainingData, model)
trainingData.shuffle()

# for example

print "Preparing data"
vectors = []
labels = []
for row in trainingData.data:
    vectors.append(row[2])
    labels.append(row[1])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(vectors, labels, test_size=0.2, random_state=0)

svm = joblib.load("../classifier/svm.classifier")
lg = joblib.load("../classifier/lg.classifier")
gnb = joblib.load("../classifier/naive_bayes.classifier")

eclf = EnsembleClassifier(clfs=[svm, lg, gnb], voting='soft')

for clf, label in zip([svm, lg, gnb, eclf], ['SVM', 'Logistic Regression', 'Naive Bayes', 'Ensemble']):

    scores = cross_validation.cross_val_score(clf, X_test, y_test, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

print classification_report(y_test, eclf.predict(X_test))

