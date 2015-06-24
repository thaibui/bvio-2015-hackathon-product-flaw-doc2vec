from gensim.models.doc2vec import Doc2Vec, infer_vector_dm

inModelName = "../model/review.doc2vec"
epochs = 20

model = Doc2Vec.load(inModelName)

print "Begin inference using the model %s" % inModelName

vector = infer_vector_dm(model, ["the", "toaster", "didn't", "last", "two", "years", "not", "quick", "toast"], steps=epochs)

print "Similar paragraph:"

print "Inference vector is %s" % vector
