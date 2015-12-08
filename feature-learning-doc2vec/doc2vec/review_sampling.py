import random

inFile = "../data/review/review-extracted.tsv"
outFile = "../data/review/dict_subsampled_400k-1.txt"
samplingPercentage = 10
outFile = open(outFile, 'w')

offset = 400000
total = 400000
xx = 0
pf = 0
print 'Begin sampling %d rows of the population in %s to %s' % (total, inFile, outFile)
for uid, line in enumerate(open(inFile)):
    if uid >= offset:
        columns = line.split("\t")
        label = columns[1]

        if label == "XX" and xx <= (total / 2):
            xx += 1
            outFile.write(line)

        if label == "PF" and pf <= (total / 2):
            pf += 1
            outFile.write(line)
    #
    # if random.randint(1, 100) <= samplingPercentage:
    #     outFile.write(line)
print 'Done sampling.'
