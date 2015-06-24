import random

inFile = "../data/review/review-sample-extracted.tsv"
outFile = "../data/review/dict_subsampled.txt"
samplingPercentage = 10
outFile = open('../data/review/dict_subsampled.txt', 'w')

print 'Begin sampling %d percent of the population in %s to %s' % (samplingPercentage, inFile, outFile)
for uid, line in enumerate(open(inFile)):
    if random.randint(1, 100) <= samplingPercentage:
        outFile.write(line)
print 'Done sampling.'
