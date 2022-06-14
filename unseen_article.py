import os #readfile
import textract #pdftotext
import spacy #import library
import re
import os #readfile
import textract #pdftotext
import re
import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import spacy #import library
import collections
from collections import Counter
from spacy.matcher import Matcher #import matcher
import pandas as pd
from nltk import word_tokenize

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from spacy.tokens import Doc, DocBin

def convertpdfstotext(path):
    filelist = []

    for fp in os.listdir(path):
        allfiles = filelist.append(os.path.join(path, fp))

    for f in filelist:
        doc = textract.process(f)
    return doc

def preprocessing(convertpdftotext):
    text = doc.decode('utf8') #convert to byte
    removing = text.replace('\n','') #remove the production of textract
    removing = text.replace('\r','') #remove the production of extract
    sentence=str(removing)
    sentence=sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    referenceremoval = text.partition("References")[0]
    return " ".join(filtered_words)

# path = r'C:\Users\shali\OneDrive\Desktop\wiki\Terminology\Unseen'
path = r'/mnt/c/Users/shali/OneDrive/Desktop/wiki/Terminology/Related_article'
doc = convertpdfstotext(path)
cleandata = preprocessing(doc)
# print(cleandata)
# This changes the result datatype to spacy.tokens.doc
nlp = spacy.load('en_core_web_sm')
data = nlp(cleandata)
# print(data)
# print(type(data))

nlp = spacy.blank("en")
doc_bin = DocBin()
doc_bin.add(data)
doc_bin.to_disk("./dev.spacy")