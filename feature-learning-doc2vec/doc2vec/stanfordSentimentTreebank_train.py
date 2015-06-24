from gensim_infer_vector.gensim.models.doc2vec import Doc2Vec, LabeledSentence
from numpy.random import shuffle

class StanfordLineSentence(object):
    lines = []
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        if len(self.lines) == 0:
            for uid, line in enumerate(open(self.filename)):
                words = line.split()
                (lastWord, sentenceId) = words.pop().split('|')
                words.append(lastWord)
                sentence = LabeledSentence(words, labels=[sentenceId])
                self.lines.append(sentence)
                yield sentence
        else:
            for line in self.lines:
                yield line

    def shuffle(self):
        shuffle(self.lines)

#fileName = "../data/stanfordSentimentTreebank/dictionary.txt"
fileName = "../data/stanfordSentimentTreebank/dictionary_subsampled.txt"
outModelName = "../model/stanfordSentimentTreebank.doc2vec"
epochs = 20
alpha = 0.025

sentences = StanfordLineSentence(fileName)
model = Doc2Vec(workers=8, size=300, alpha=alpha)

print "Begin building the vocabulary on %s" % fileName

model.build_vocab(sentences)

for epoch_count in range(epochs):
    print "Training epoch #%d " % epoch_count
    sentences.shuffle()
    model.alpha -= (alpha/epochs)
    model.min_alpha = model.alpha
    model.train(sentences)
    print "Current learning rate %f" % model.alpha

print "Done. Saving model to %s" % outModelName

model.save(outModelName)
