import random

inFile = "../data/review/review-extracted.tsv"
outFile = "../data/review/dict_subsampled_20k-4.txt"
samplingPercentage = 10
outFile = open(outFile, 'w')

offset = 60000
total = 20000
xx = 0
pf = 0
print 'Begin sampling %d percent of the population in %s to %s' % (samplingPercentage, inFile, outFile)
for uid, line in enumerate(open(inFile)):
    if uid >= offset:
        columns = line.split("\t")
        label = columns[1]

        if label == "XX" and xx <= total:
            xx += 1
            outFile.write(line)

        if label == "PF" and pf <= total:
            pf += 1
            outFile.write(line)
    #
    # if random.randint(1, 100) <= samplingPercentage:
    #     outFile.write(line)
print 'Done sampling.'
