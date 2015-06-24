from gensim_infer_vector.gensim.models.doc2vec import Doc2Vec, infer_vector_dm
from numpy.random import shuffle

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
                # id, label, vectors
                row = (columns[0], columns[1], doc2vec[id], columns[3], columns[4], columns[5])
                self.data.append(row)
            else:
                print "WARNNING: Can't find ID %s " % id
        print "Vectorized Data loaded containing %d rows" % len(self.data)

    def shuffle(self):
        shuffle(self.data)

inTrainingData = "../data/review/dict_subsampled.txt"
inModelName = "../model/review.doc2vec"
epochs = 20

# Load the model
model = Doc2Vec.load(inModelName)

# Prepare training data
trainingData = VectorizedData(inTrainingData, model)

# for example
print trainingData.data[0]
