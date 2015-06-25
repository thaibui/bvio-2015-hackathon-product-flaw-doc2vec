from gensim_infer_vector.gensim.models.doc2vec import Doc2Vec, infer_vector_dm
from numpy.random import shuffle
from sklearn.externals import joblib

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

filename = "../reviews/review-sample-extracted-walmart-unlabeled.tsv"
inModelName = "../model/20k-2_500_40_dbow_negative15_window16.doc2vec"
epochs = 40

model = Doc2Vec.load(inModelName)

lg = joblib.load("../classifier/lg.classifier")
pfCount = 0
for uid, line in enumerate(open(filename)):
    if uid % 1000 == 0:
        print "Scanning line %d for Product Flaws, found %d" % (uid, pfCount)
    columns = line.split("\t")
    text = columns[2]
    id = columns[0]
    words = []
    # sanitize the words
    for w in text.split(" "):
        r = w.lower()
        t = ''
        for c in r:
            if c != "." and c != "," and c != "!":
                t += c
        words.append(t)

    vector = infer_vector_dm(model, words, steps=epochs)
    if lg.predict(vector) == "PF":
        pfCount += 1
        print "Product Flaws:"
        print "Review ID: %s" % id
        print "Review Text:\n==\n%s\n==\n" % text
