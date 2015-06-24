from gensim.models.doc2vec import Doc2Vec, infer_vector_dm

inModelName = "../model/stanfordSentimentTreebank.doc2vec"
epochs = 20

model = Doc2Vec.load(inModelName)

print "Begin inference using the model %s" % inModelName

vector = infer_vector_dm(model, ["movie", "high", "ping-poing"], steps=epochs)

print "Done. Inference vector is %s" % vector
