from gensim_infer_vector.gensim.models.doc2vec import Doc2Vec
from numpy.random import shuffle
from sklearn import svm, cross_validation, linear_model
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from mlxtend.sklearn import EnsembleClassifier
import pandas

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

inTrainingData = "../data/review/dict_subsampled_1k-4.txt"
inModelName = "../model/1k-4_500_40_dbow_negative15_window16.doc2vec"

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

df = pandas.DataFrame(columns=('w1', 'w2', 'w3', 'mean', 'std'))
i = 0
for w1 in range(1, 4):
    for w2 in range(1, 4):
        for w3 in range(1, 4):
            if len(set((w1, w2, w3))) == 1:
                continue

            eclf = EnsembleClassifier(clfs=[svm, lg, gnb], voting='soft', weights=[w1, w2, w3])
            scores = cross_validation.cross_val_score(
                estimator=eclf,
                X=X_test,
                y=y_test,
                cv=5,
                scoring='accuracy',
                n_jobs=1)
            df.loc[i] = [w1, w2, w3, scores.mean(), scores.std()]
            i += 1

df.sort(columns=['mean', 'std'], ascending=False)
print df
