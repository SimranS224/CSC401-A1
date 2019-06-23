import numpy as np
import sys
import argparse
import os
import json
import re
import copy
import string
import csv
# files 
BGLnounsunique1 = csv.reader(open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r'))
Warringernormsunique1 = csv.reader(open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', 'r'))

altunique1 = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')
rightunique2 = np.load('/u/cs401/A1/feats/Right_feats.dat.npy')
centreunique3 = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')
leftunique4 = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
# text files 
alttextlist1 = [line.rstrip() for line in open("/u/cs401/A1/feats/Alt_IDs.txt", "r")]
lefttextlist2 = [line.rstrip() for line in open("/u/cs401/A1/feats/Left_IDs.txt", "r")]
righttextlist3 = [line.rstrip() for line in open("/u/cs401/A1/feats/Right_IDs.txt", "r")]
centretextlist4 = [line.rstrip() for line in open("/u/cs401/A1/feats/Center_IDs.txt", "r")]


def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # the numpy array of the 29 features 
    my_array = np.zeros([29], dtype=float)

    # first person pronouns
    firstppossiblewords = ['(?<![^\s])I/', '(?<![^\s])me/', '(?<![^\s])my/+', '(?<![^\s])mine/', '(?<![^\s])we/', '(?<![^\s])us/', '(?<![^\s])our/', '(?<![^\s])ours/']

    numfirstppronoun = 0
    for x in firstppossiblewords:
       
        numfirstppronoun += len(re.findall(x, comment, re.IGNORECASE))

    my_array[0] = numfirstppronoun

    # second person pronouns
    secondp_possible_words = ['(?<![^\s])you/', '(?<![^\s])your/', '(?<![^\s])yours/', '(?<![^\s])u/', '(?<![^\s])ur/', '(?<![^\s])urs/']
    numsecondppronoun = 0
    for x in secondp_possible_words:
        
        numsecondppronoun += len(re.findall(x, comment,re.IGNORECASE))
       

    my_array[1] = numsecondppronoun

    # third person pronoun
    thirdp_possiblewords = ['(?<![^\s])he/', '(?<![^\s])him/', '(?<![^\s])his/', '(?<![^\s])she/', '(?<![^\s])her/', '(?<![^\s])hers/', '(?<![^\s])it/', '(?<![^\s])its/', '(?<![^\s])they/', '(?<![^\s])them/', '(?<![^\s])their/', '(?<![^\s])theirs/']
    numthirdppronoun = 0
    for x in thirdp_possiblewords:
       
        numthirdppronoun += len(re.findall(x, comment,re.IGNORECASE))
    my_array[2] = numthirdppronoun
        

    # num coordinating conjunctions
    coordinatingconjwords = ['CC']
    numcoordinatingconjwords = 0
    for x in coordinatingconjwords:
        
        numcoordinatingconjwords += len(re.findall(x, comment,re.IGNORECASE))
        
    my_array[3] = numcoordinatingconjwords

    # past tense verbs
    pasttenseverbs = ["VBD"]
    numpasttenseverbs = 0
    for x in pasttenseverbs:
        
        numpasttenseverbs += len(re.findall(x, comment))
      
    my_array[4] = numpasttenseverbs

    #future tense verbs 
    futuretenseverbs = [r"’ll/", r"(?<![^\s])will/", r"(?<![^\s])gonna/", r'go\sto[.+]/VB']
    numfuturetenseverbs = 0
    for x in futuretenseverbs:
       
        numfuturetenseverbs += len(re.findall(x, comment,re.IGNORECASE))
    my_array[5] = numfuturetenseverbs
    
    # commas
    numcommas = 0
    commas = [',']
    for x in commas:
        
        numcommas += len(re.findall(x,comment,re.IGNORECASE))
   
    my_array[6] = numcommas
    
    # Number of multi-character punctuation tokens
    nummultline = 0
    allpunctuationone = r"[!#\$%&\\\(\)\*\+,\-\.:;<=>\?@\[\\\]\^_`{\|}~']{2,}" # check with tony
    # print(re.findall(allpunctuationone,comment))
    nummultline += len(re.findall(allpunctuationone,comment))

    # print(nummultline)
    my_array[7] = nummultline



    # common nouns
    numcommonnouns = 0
    commonnouns = ['NN', 'NNS']
    for x in commonnouns:
        # print("numcommonnouns")
        # print(re.findall(x, comment, re.IGNORECASE))
        numcommonnouns += len(re.findall(x,comment))
    my_array[8] = numcommonnouns
    # print("the number of numcommonnouns is " + str(numcommonnouns))

    # propernouns
    numpropernouns =0
    propernouns = ['NNP', 'NNPS']
    for x in propernouns:
        # print("numpropernouns")
        # print(re.findall(x, comment, re.IGNORECASE))
        numpropernouns += len(re.findall(x,comment,re.IGNORECASE))
    # print("the number of numpropernouns is " + str(numpropernouns))
    my_array[9] = numpropernouns


   # adverbs
    numadverbs = 0
    adverbs = ['RBS','RB', 'RBR']
    for x in adverbs:
        # print("numadverbs")
        # print(re.findall(x, comment, re.IGNORECASE))
        numadverbs += len(re.findall(x,comment,re.IGNORECASE))
    # print("the number of numadverbs is " + str(numadverbs))
    my_array[10] = numadverbs



    #wh-words
    numwh_words = 0
    wh_words = ['WP','WDT', 'WP$', 'WRB']
    for x in wh_words:
        # print("numwh_words")
        # print(re.findall(x, comment, re.IGNORECASE))
        numwh_words += len(re.findall(x,comment))
    # print("the number of numwh_words is " + str(numwh_words))
    my_array[11] = numwh_words


    # slang acronyms
    numslang_acronyms = 0
    slang_acronyms = ['lms','f2f', 'gtr','bw', 'imho', 'tbh', 'rofl', 'wtf','ru', 'so', 'tc', 'tmi',
                      'ym','bff', 'wyd', 'lylc', 'brb','smh', 'fwb', 'lmfao', 'lmao','atm', 'imao',
                      'sml', 'btw', 'ttyl', 'imo', 'ltr','thx', 'kk', 'omg', 'ttys', 'afn', 'bbs',
                      'cya', 'ez', 'fyi', 'ppl','sob', 'ic', 'jk', 'k', 'ly', 'ya', 'nm', 'np',
                      'plz', 'ur', 'u', 'sol', "fml", "lol"]
    for x in slang_acronyms:
        # print("numslang_acronyms")
        # print(re.findall(x, comment, re.IGNORECASE))
        numslang_acronyms += len(re.findall("(?<![^\s])" + x + "/",comment, re.IGNORECASE))
    # print("the number of numslang_acronyms is " + str(numslang_acronyms))
    my_array[12] = numslang_acronyms


    # uppercase
    numuppercaseletters = 0
    copyofcomment = copy.copy(comment)
    thelistwords = copyofcomment.split()
    for x in thelistwords:
        # print("numuppercaseletters")
        actualword = x[:x.find("/")]
        actualword.isupper()
        if len(x) >= 3 and actualword.isupper():
            # print(x)
            numuppercaseletters += 1
    # print("the number of numuppercaseletters is " + str(numuppercaseletters))
    my_array[13] = numuppercaseletters

    # avglengthofsentencestoken
    avglengthofsentences = 0
    for x in thelistwords:
        # print("avglengthofsentences")
        # print(x)
        avglengthofsentences += len(x)
    # print("the number of avglengthofsentences is " + str(avglengthofsentences))
    my_array[14] = avglengthofsentences


    # Average length of tokens, excluding punctuation-only tokens, in characters
    avglenghtignorepunc = 0
    allpunctuation = list(string.punctuation)

    newstring = ''
    for x in comment:
        puncsmatch = []
        for p in allpunctuation:
            check = x == p
            if check:
                puncsmatch.append(x)
        # print(puncsmatch)
        if len(puncsmatch) == 0:
            newstring += x
    # print(" the content of newstring" + newstring)
    splitednewstring = newstring.split(" ")
    for x in splitednewstring:
        avglenghtignorepunc += len(x)
    # print("the number of not puncs is " + str(avglenghtignorepunc))
    my_array[15] = avglenghtignorepunc


    # num sentences
    sentencecopy = copy.copy(comment)
    splitedsentences= list(filter(bool, sentencecopy.splitlines()))
    numsentences = len(splitedsentences)

    my_array[16] = numsentences
    # print("the number of not numsentences is " + str(numsentences))


    # 18 - 23. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    sum = 0
    count = 0

    # Average of FAM from Bristol, Gilhooly, and Logie norms
    secondsum = 0
    secondcount = 0

    # Average of FAM from Bristol, Gilhooly, and Logie norms
    thirdsum = 0
    thirdcount = 0

    # arrays of values associated with corresponding columns from warringer nouns
    AoA = []
    IMG = []
    FAM = []
    # copy it to manipulate the contents 
    copyofcomment2 = copy.copy(comment)
    splittedcopy = copyofcomment2.split()
    for row in BGLnounsunique1:
        for token in splittedcopy:
            # print(row[1])
            # print(type(row[1]))
            # print(token)
            amatch = re.findall("(?<![^\s])" + row[1] + "/",token)
            if str(row[3]) != "AoA (100-700)" and str(row[3]) != "" and amatch != []:
                count += 1
                # print(row[3])
                sum += float((row[3]))
                AoA.append((row[3]))
            if str(row[4]) != "IMG"and str(row[4]) != "" and amatch != []:
                secondcount += 1
                # print(row[4])

                secondsum += float((row[4]))
                IMG.append((row[4]))
            if str(row[5]) != "FAM" and str(row[5]) != "" and amatch != []:
                thirdcount += 1
                # print(row[5])
                thirdsum += float((row[5]))
                FAM.append((row[5]))
    # conver to numpy array to perform mean and std on 
    AoAasnp = np.array(AoA).astype(np.float)
    IMGasnp = np.array(AoA).astype(np.float)
    FAMasnp = np.array(FAM).astype(np.float)
    # default to 0 , then calculate if there are values associated with the column
    avgFam = 0
    avgAOA = 0
    avgIMG = 0
    stdFAM = 0
    stdAoA = 0
    stdIMG = 0
    if (AoAasnp.size != 0):
        avgAOA = AoAasnp.mean()
        stdAoA = AoAasnp.std()
    if (IMGasnp.size != 0):
        avgIMG = IMGasnp.mean()
        stdIMG = IMGasnp.std()
    if (FAMasnp.size != 0):
        avgFam = FAMasnp.mean()
        stdFAM = FAMasnp.std()

    # add it 
    my_array[17] = avgIMG
    my_array[18] = avgFam
    my_array[19] = avgAOA
    my_array[20] = stdAoA
    my_array[21] = stdIMG
    my_array[22] = stdFAM



    # 24. Average of V.Mean.Sum from Warringer norms
    # 25. Average of A.Mean.Sum from Warringer norms
    # 26. Average of D.Mean.Sum from Warringer norms
    # 27. Standard deviation of V.Mean.Sum from Warringer norms
    # 28. Standard deviation of A.Mean.Sum from Warringer norms
    # 29. Standard deviation of D.Mean.Sum from Warringer norms
    # variable setup
    vmeansum = 0
    vmeancount = 0
    vmean = []
    ameansum = 0
    ameancount = 0
    amean = []
    dmeansum =0
    dmeancount =0
    dmean = []
    # copy it to manipulate the contents 

    copyofcomment3 = copy.copy(comment)
    splittedcop3 = copyofcomment3.split()
    for row in Warringernormsunique1:
        for token in splittedcop3:
            # print(row[1])
            # print(type(row[1]))
            # print(token)
            amatch = re.findall("(?<![^\s])" + row[1] + "/", token)
            if str(row[2]) != "V.Mean.Sum" and str(row[2]) != "" and amatch != []:
                vmeancount += 1
                # print(row[2])
                vmeansum += float((row[2]))
                vmean.append((row[2]))
            if str(row[5]) != "A.Mean.Sum" and str(row[5]) != "" and amatch != []:
                ameancount += 1
                # print(row[4])

                ameansum += float((row[5]))
                amean.append((row[5]))
            if str(row[9]) != "D.Mean.Sum" and str(row[9]) != "" and amatch != []:
                dmeancount += 1
                # print(row[5])
                dmeansum += float((row[9]))
                dmean.append((row[9]))
    # conver to numpy array to perform mean and std on 

    vmeanasnp = np.array(vmean).astype(np.float)
    ameanasnp = np.array(amean).astype(np.float)
    dmeanasnp = np.array(dmean).astype(np.float)

    avgvmean = 0
    avgamean = 0
    avgdmean = 0
    stdvmean = 0
    stdamean = 0
    stddmean = 0
    if (vmeanasnp.size != 0):
        avgvmean = vmeanasnp.mean()
        stdvmean = vmeanasnp.std()
    if (ameanasnp.size != 0):
        avgamean = ameanasnp.mean()
        stdamean = ameanasnp.std()
    if (dmeanasnp.size != 0):
        avgdmean = dmeanasnp.mean()
        stddmean = dmeanasnp.std()
    # add the calulated values 

    my_array[23] = avgvmean
    my_array[24] = avgamean
    my_array[25] = avgdmean
    my_array[26] = stdvmean
    my_array[27] = stdamean
    my_array[28] = stddmean

    return my_array














def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))
    extractfeaturesarray = np.zeros(29)


    list_ = []  
    i = 0
    for line in data:  
        thecurrentrow = 1
        # " (0: Left, 1: Center, 2: Right, 3: Alt)"
        # the category
        thecat = 0
        # read json to loop over 
        line = json.loads(line)
        # the o LIWC/Receptiviti features ones 
        otherfeaturesarray = [0] * 144
        if (line["cat"] == "Right"):
            thecat = 2
            thecurrentrow = int(righttextlist3.index(line["id"]))
            otherfeaturesarray = list(np.array(rightunique2[thecurrentrow]).tolist())
        elif line['cat'] == "Left":
            thecat = 0
            thecurrentrow = int(lefttextlist2.index(line["id"]))
            otherfeaturesarray = list(np.array(leftunique4[thecurrentrow]).tolist())
        elif line['cat'] == "Center":
            thecat = 1
            thecurrentrow = int(centretextlist4.index(line["id"]))
            otherfeaturesarray = list(np.array(centreunique3[thecurrentrow]).tolist())
        elif line["cat"] == "Alt":
            thecat = 3
            thecurrentrow = int(alttextlist1.index(line["id"]))
            otherfeaturesarray = list(np.array(altunique1[thecurrentrow]).tolist())
        # get the features from the above function
        extractfeaturesarray = extract1(line["body"])
        # efaasarray is the extracted features as a list so I can concatenate 
        #it with the other LIWC/Receptiviti features
        efaasarray = list(np.array(extractfeaturesarray).tolist())
        # print(efaasarray)
        # print(len(efaasarray))
        efaasarray.extend(otherfeaturesarray)
        efaasarray.append(thecat)
        finalndarray = np.asarray(efaasarray)
        # print(finalndarray)
        # print(finalndarray[100])
        # i indexes the row to add the 174 features long data
        feats[i] =finalndarray
        i += 1
   

    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

