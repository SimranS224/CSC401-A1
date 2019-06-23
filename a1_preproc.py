import sys
import argparse
import os
import json
import re
import spacy
import html
import copy
import json
# indir = '/u/cs401/A1/data/';s
indir = '/u/cs401/A1/data/'
stopwordsfileunique1 = '/u/cs401/Wordlists/StopWords'
abbrevunique2 = '/u/cs401/Wordlists/abbrev.english'
pn_abbrevunique3 = '/u/cs401/Wordlists/pn_abbrev.english'

nlp = spacy.load('en', disable = ['parser', 'ner'])

def preproc1(comment, steps=range(1, 11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''
    modComm = comment
    if 1 in steps:
        # Remove all newline characters
        newline = re.compile('\n')
        modComm = re.sub(newline, '', modComm)
    if 2 in steps:
        #Replace HTML character codes
        html.unescape(modComm)
    if 3 in steps:
        # Remove all URLs
        modComm = re.sub('http\S+','',modComm)
        modComm = re.sub('www\S+','',modComm)
    if 4 in steps:
        #Split each punctuation
        newstring = ''
        punctuation = re.compile('[^\w]+')
        for i in range(len(modComm)):
            check = re.findall(punctuation, modComm[i])
            if check == [] or modComm[i] == ' ':
                newstring += modComm[i]
            else:
                newstring += ' '
                newstring += modComm[i]
                newstring += ' '
        modComm = newstring.rstrip()


    if 5 in steps:
        #. Split clitics using whitespace
        list_of_clitics = ["\'d", "\'n", "\'ve", "\'re", "\'ll", "\'m", "\'re", "\'s", "t\'", "y\'", "\'t"]
        splitted = modComm.split(' ')
        newarray = []
        for word in splitted:
            if word[-3:] in list_of_clitics:
                word = word[:word.find(word[-3:])] + ' ' + word[-3:]
                newarray.append(word)
            elif word[-2:] in list_of_clitics:
                word = word[:word.find(word[-2:])] + ' ' + word[-2:]
                newarray.append(word)
            elif len(word) > 2 and word[-1] == "'" and word[-2] == "s":
                word = word[:-1] + " " + word[-1]
                newarray.append(word)
            else:
                newarray.append(word)
        modComm = " ".join(newarray)
    if 6 in steps:
        #Each token is tagged with its part-of-speech using spaCy
        utt = nlp(modComm)
        nstring = ''
        for token in utt:
            if (token.text) == ".":
                nstring += str(token) + "/" + " "

            else:
                nstring += str(token) + "/" + str(token.tag_) + " "
        modComm = nstring.rstrip()
    if 7 in steps:
        # Remove stopwords
        listabbrev = []
        # c = os.getcwd() # change to absolute path
        # # stopwordsfile = '../wordlists/StopWords'
        # fullFile = os.path.join(c, stopwordsfileunique1)
        # print(fullFile)
        data = open(stopwordsfileunique1, 'r')
        for line in data:
            linecopy = copy.copy(line)
            linecopy = re.sub(r'\n', '', linecopy)
            listabbrev.append(linecopy)
        nstring = ""
        splitted = modComm.split(' ')
        for word in splitted:
            actualword = word[:word.find("/")]
            if actualword not in listabbrev:
                nstring += word + " "
        modComm = nstring.rstrip()
    if 8 in steps:
        #Apply lemmatization using spaCy
        orginalmodComm = copy.copy(modComm)
        narray = []
        splitted = orginalmodComm.split(' ')
        for word in splitted:
            index = (word.find("/"))
            actualword = word[:index]
            narray.append(actualword)
        utt = nlp(modComm)
        nstring = ''
        for token in utt:
            if str(token) in narray or token.text == "./":
                if token.text == "./":
                    nstring += str(token.lemma_) + " "
                elif not str(token).startswith("-"):
                   
                    str(token).replace(str(token), token.lemma_)
                    nstring += str(token.lemma_) + "/" + str(token.tag_) + " "
                else:
                    nstring += str(token) + "/" + str(token.tag_) + " "
        modComm = nstring
    if 9 in steps:
        #Add a newline between each sentence.
        listabbrev = []
        # c = os.getcwd()
        # fullFile = os.path.join(c, abbrevunique2)
        # print(fullFile)
        data = open(abbrevunique2, 'r')
        for line in data:
            linecopy = copy.copy(line)
            linecopy = re.sub(r'\n', '', linecopy)
            listabbrev.append(linecopy)
        listPNabbrev = []
        # d = os.getcwd()
        # fullFile = os.path.join(d, pn_abbrevunique3)
        # # print(fullFile)
        data = open(pn_abbrevunique3, 'r')
        for line in data:
            linecopy = copy.copy(line)
            linecopy = re.sub(r'\n', '', linecopy)
            listPNabbrev.append(linecopy)
        indexesofperiods = []
        curindex = -1
        for i in range(len(modComm)):
            if modComm[i] == ".":
                indexesofperiods.append(i)
            i+= 1
        
        combinedlist = listPNabbrev + listabbrev
        if len(indexesofperiods) != 0:

            count = -2
            newstring = ''
            # print(listabbrev)
            for index in indexesofperiods:
                # print("in loop")
                if modComm[index-2:index+1] in combinedlist or  modComm[index-3:index+1] in combinedlist or  modComm[index-3:index+1] in combinedlist:
                    pass
                else:
                    # print("new sentence" + newstring)
                    newstring += modComm[count+2:index+1] + " "+ "\n"
                    count = index
            modComm = newstring


    if 10 in steps:
        # Convert text to lowercase
        splitted = modComm.split(' ')
        totalstring = ""
        for word in splitted:
            newword = ""
            tomakeupper = word[:word.rfind("/")]
            other = word[word.find("/"):]
            newword += (tomakeupper.lower() + other)
            totalstring += newword + " "
        modComm = totalstring.rstrip()
    if modComm == "/":
        modComm = ""
    return modComm


# features to remove from dataset 
to_remove = ["archived", "retrieved_on", "author_flair_text",
             "author", "score", 'created_utc', "parent_id", 'score_hidden', "edited", "subreddit_id",
             "distinguished", "gilded", "link_id", "author_flair_css_class", "stickied", ]


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))
            max = args.ID[0]
            # print("the value ")
            # print(args.max)
            list_ = []  
            # this chunk of code gets the correct random sample from the dataset 
            if (10000 + (args.ID[0] % len(data))) > len(data):
                extra_needed = ((10000 + args.ID[0] % len(data)) - args.ID[0] % len(data))
                my_slice = data[args.ID[0] % len(data):] + data[:extra_needed]
            else:
                my_slice = data[args.ID[0] % len(data): 10000 + args.ID[0] % len(data)]

            for line in my_slice:  
                list_.append(json.loads(line))

            for line in list_:
                all_keys = []
                # removes uneccary keys
                for thing in to_remove:
                    if thing in line:
                        del line[thing]
                # adds all keys of the current line to a list to use for filtering late
                line["cat"] = file

            for line in list_:
                all_keys = []
                for element in line:
                    all_keys.append(element)
                for key in all_keys:
                    if key in line:
                        if line[key] == "[deleted]":  # filters all with deleted
                            list_.remove(line)
                line["body"] = preproc1(line["body"])
                # allOutput.append((str(line)))
                # adds to a json with the keys in sorted order 
                allOutput.append(json.dumps(line, sort_keys=True, indent=4))
            print("output")
            print(list_[0])
           
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
