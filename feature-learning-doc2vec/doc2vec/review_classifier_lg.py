from gensim_infer_vector.gensim.models.doc2vec import Doc2Vec
from numpy.random import shuffle
from sklearn import cross_validation, linear_model
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pickle

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

inTrainingData = "../data/review/dict_subsampled_20k-2.txt"
inModelName = "../model/20k-2_500_40_dbow_negative15_window16.doc2vec"

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

print "Fitting the model on %d vectors" % len(X_train)

classifier = linear_model.LogisticRegression(C=1.0)
classifier.fit(X_train, y_train)

print "Evaluate the model"

print "Test dataset %d " % len(X_test)

print "Mean accuracy %f " % classifier.score(X_test, y_test)
print "F1-score: "
print classification_report(y_test, classifier.predict(X_test))


joblib.dump(classifier, "../classifier/lg.classifier")
