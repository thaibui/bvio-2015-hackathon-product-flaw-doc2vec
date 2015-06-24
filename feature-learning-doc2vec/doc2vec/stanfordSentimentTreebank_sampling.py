import random

inFile = "../data/stanfordSentimentTreebank/dictionary.txt"
samplingPercentage = 20
outFile = open('../data/stanfordSentimentTreebank/dictionary_subsampled.txt', 'w')

print 'Begin sampling %d of the population in %s to %s' % (samplingPercentage, inFile, outFile)
for uid, line in enumerate(open(inFile)):
    if random.randint(1, 100) <= samplingPercentage:
        outFile.write(line)
print 'Done sampling.'
