from gensim_infer_vector.gensim.models.doc2vec import Doc2Vec, infer_vector_dm
from sklearn.externals import joblib
import sys

inModelName = "../model/20k-1_500_40_dbow_negative15_window16.doc2vec"
epochs = 40

model = Doc2Vec.load(inModelName)

lg = joblib.load("../classifier/svm.classifier")

text = sys.argv[1]
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
    print "Product Flaws"
else:
    print "Normal"
