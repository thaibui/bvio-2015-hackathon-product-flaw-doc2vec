from gensim_infer_vector_cp.gensim.models.doc2vec import Doc2Vec, LabeledSentence
from numpy.random import shuffle

class ReviewLineSentence(object):
    lines = []
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        if len(self.lines) == 0:
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

                sentence = LabeledSentence(words, labels=[columns[0]])
                self.lines.append(sentence)
                yield sentence
        else:
            for line in self.lines:
                yield line

    def shuffle(self):
        shuffle(self.lines)

#fileName = "../data/stanfordSentimentTreebank/dictionary.txt"
fileName = "../data/review/dict_subsampled.txt"
outModelName = "../model/review.doc2vec"
epochs = 10
alpha=0.025

sentences = ReviewLineSentence(fileName)
model = Doc2Vec(workers=8, size=100, alpha=alpha)

print "Begin building the vocabulary on %s" % fileName

model.build_vocab(sentences)

for epoch_count in range(epochs):
    print "Training epoch #%d " % epoch_count
    # sentences.shuffle()
    model.alpha -= (alpha/epochs)
    model.min_alpha = model.alpha
    model.train(sentences)
    print "Current learning rate %f" % model.alpha

print "Done. Saving model to %s containing %d vocabularies" % outModelName, len(model.vocab)

model.save(outModelName)
