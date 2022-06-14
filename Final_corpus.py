import os #readfile
import numpy as np
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
from spacy.tokens import Doc, DocBin
def convertpdfstotext(path):
    filelist = []

    for fp in os.listdir(path):
        allfiles = filelist.append(os.path.join(path, fp))

    for f in filelist:
        doc = textract.process(f)
    return doc
path = r'/mnt/c/Users/shali/OneDrive/Desktop/wiki/Terminology/Terminology_project/Data'
doc = convertpdfstotext(path)
# print(doc)
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

cleandata = preprocessing(doc)
nlp = spacy.load('en_core_web_sm') #load model which is sm small model and en means english
data = nlp(cleandata)

matcher = Matcher(nlp.vocab) # Initialize the matcher with the shared vocab

#define rules
pattern1 = [{'POS': 'ADJ'},{'POS': 'NOUN'}]
pattern2 = [{'POS': 'NOUN'}, {'POS': 'NOUN'}]
pattern3 = [{'POS': 'NOUN'}, {'POS': 'NOUN'}, {'POS': 'NOUN'} ]
pattern4 = [{'POS': 'NOUN'}]
pattern5 = [{'POS': 'ADJ'},{'POS': 'NOUN'}, {'POS': 'NOUN'}]

#add rules to matcher
matcher.add('ADJ+N', [pattern1])
matcher.add('N+N', [pattern2])
matcher.add('N+N+N', [pattern3])
matcher.add('N', [pattern4])
matcher.add('ADJ+N+N', [pattern5])

matches = matcher(data)

d=[]
for match_id, start, end in matches:
    rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
    span = data[start : end]  # get the matched slice of the doc
    d.append((rule_id, span.text))
    keyterm = span.text

total_terms = Counter(d)
rank_term = total_terms.most_common()
df = pd.DataFrame(rank_term, columns=['terms', 'frequency'])
# df.to_csv('first_time_keyterm_extraction.csv', encoding='utf-8')
p1 = [{'LOWER': 'encoder'}, {'IS_PUNCT': True, 'OP':'?'}, {'LOWER': 'decoder'}]
p2 = [{"POS": {"IN": ["NOUN", "ADJ"]}}, {'LOWER': 'encoder'}]
p3 = [{"POS": {"IN": ["NOUN", "ADJ"]}}, {'LOWER': 'decoder'}]
p4 = [{'LOWER': 'encoder'}, {'IS_PUNCT': True}, {'LOWER': 'decoder'},{"POS": {"IN": ["NOUN"]}}]
p5 = [{'LOWER': 'encoder'}]
p6 = [{'LOWER': 'decoder'}]
p7 = [{'LEMMA': 'encoder'}]
p8 = [{'LEMMA': 'decoder'}]
p9 = [{'LOWER': 'token'}]
p10 = [{'LEMMA': 'token'}]
p11 = [{'LOWER': 'infobox'}]
p12 = [{'LOWER': 'info'}, {'IS_PUNCT': True, 'OP':'?'} , {'LOWER': 'box'}]
p13 = [{'LOWER': 'machine'}, {'LOWER': 'translation'},{'POS': 'NOUN','OP':'?'}]
p14 = [{"POS": {"IN": ["NOUN", "ADJ"]}}, {'LOWER': 'machine'}, {'LOWER': 'translation'}]
p15 = [{'LOWER': 'nlg'}]
p16 = [{'LOWER': 'natural'},{'IS_PUNCT': True}, {'LOWER': 'language'},{'IS_PUNCT': True}, {'LOWER': 'generation'}]
p17 = [{'LOWER': 'neural'}, {'LOWER': 'network'},{'POS': 'NOUN','OP':'?'}]
p18 = [{"POS": {"IN": ["NOUN", "ADJ"]}}, {'LOWER': 'neural'}, {'LOWER': 'network'}]
p19 = [{"POS": {"IN": ["NOUN", "ADJ"]}}, {'LOWER': 'embedding'}]
p20 = [{"POS": {"IN": ["NOUN", "ADJ"]}}, {'LEMMA': 'embedding'}]
p21 = [{'LOWER': 'beam'}, {'POS': 'NOUN'}]
p22 = [{'LOWER': 'ngram'}]
p23 = [{'LOWER': 'n'}, {'LOWER': 'gram'}]
p24 = [{'LOWER': 'n'},{'IS_PUNCT': True, 'OP':'?'}, {'LOWER': 'gram'}]
p25 = [{'LOWER':'generator'}]
p26 = [{'LOWER':'parser'}]
p27 = [{'LOWER':'ontology'}]
p28 = [{'LOWER':'softmax'}]
p29 = [{'LEMMA':'ontology'}]
p30 = [{'LOWER':'backprop'}]
p31 = [{'LOWER':'token'}]
p32 = [{'LOWER':'vector'}]
p33 = [{'LOWER':'array'}]
p34 = [{'LOWER':'copy'}, {'LEMMA':'action'},{'POS': 'NOUN','OP':'?'} ]
p35 = [{'LOWER':'copy'}, {'LOWER':'action'},{'POS': 'NOUN','OP':'?'} ]
p36 = [{'LOWER':'sochastic','OP':'?'}, {'LOWER':'gradient'}, {'LOWER':'descent'}]
p37 = [{'LOWER':'attention'}, {'POS': 'NOUN','OP':'?'}]
p38 = [{"POS": {"IN": ["NOUN","ADJ"]}}, {'LOWER':'attention'}]
p39 = [{"POS": {"IN": ["NOUN","ADJ","PROPN"]}},{'POS': 'NOUN','OP':'?'}, {'LOWER': 'function'}]
p40 = [{"POS": {"IN": ["NOUN","ADJ","PROPN"]}},{'POS': 'NOUN','OP':'?'}, {'LEMMA': 'function'}]
p41 = [{"POS": {"IN": ["NOUN","ADJ","PROPN"]}}, {'LOWER': 'layer'}]
p42 = [{"POS": {"IN": ["NOUN","ADJ","PROPN"]}}, {'LEMMA': 'layer'}]
p43 = [{"POS": {"IN": ["NOUN","ADJ"]}}, {'LOWER': 'search'}]
p44 = [{'POS': 'NOUN','OP':'?'}, {'LOWER':'vector'}]
p45 = [{'POS': 'NOUN','OP':'?'}, {'LEMMA':'vector'}]
p46 = [{'POS': 'COMP','OP':'?'}, {'LOWER':'knowledge'},{'IS_PUNCT': True, 'OP':'?'}, {'LOWER':'base'}]
p47 = [{'LOWER':'knowledge'},{'IS_PUNCT': True, 'OP':'?'}, {'LOWER':'bases'}]
p48 = [{'LOWER':'dialog'}, {'LOWER':'system'}, {'POS': 'NOUN','OP':'?'} ]
p49 = [{'LOWER':'dialog'}, {'LEMMA':'system'}, {'POS': 'NOUN','OP':'?'} ]
p50 = [{'LOWER':'local'}, {'LOWER':'field'}]
p51 = [{'LOWER':'global'}, {'LOWER':'field'}]
p52 = [{'LOWER':'dot'}, {'LOWER':'product'}]
p53 = [{'LOWER':'statistical'}, {'LOWER':'generation'}, {'LOWER':'model'}]
p54 = [{'LOWER':'alignment'}, {'LEMMA':'tree'}]
p55 = [{'LOWER':'lstm'}, {'POS': 'NOUN','OP':'?'}]
p56 = [{'LOWER':'linear'}, {'LOWER': 'transformation'}]
p57 = [{'POS': 'NOUN','OP':'?'}, {'LOWER':'likelihood'}, ]
p58 = [{'LOWER':'computational'}, {'LEMMA': 'linguistic'}]
p59 = [{'LOWER':'bleu'}, {'LEMMA': 'score'}]
p60 = [{'LOWER':'hyperparameter'}]
p61 = [{'LOWER':'corenlp'}]
p62 = [{'LEMMA':'hyperparameter'}]
p63 = [{'LOWER':'hyper'}, {'IS_PUNCT': True, 'OP':'?'}, {'lemma':'parameter'} ]
p64 = [{'LOWER':'content'}, {'LOWER':'selection'}, {'POS': 'NOUN','OP':'?'} ]
p65 = [{'LOWER':'sentence'}, {'LOWER':'planning'}, {'POS': 'NOUN','OP':'?'} ]
p66 = [{'LOWER':'factorization'}, {'POS': 'NOUN','OP':'?'} ]
p67 = [{'LEMMA':'weight'}, {'POS': 'NOUN','OP':'?'} ]
p68 = [{'LOWER':'space'}, {'LOWER': 'latent'}]
p69 = [{"POS": {"IN": ["NOUN","ADJ"]}},{'POS': 'NOUN','OP':'?'}, {'LEMMA':'model'}]
p70 = [{'LOWER':'surface'}, {'LOWER':'realization'}, {'POS': 'NOUN','OP':'?'} ]
p71 = [{'LOWER':'model'}, {'LOWER':'conditioning'}]

string_matcher = Matcher(nlp.vocab)

string_matcher.add('rulebased', [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,
                        p28,p29,p30,p31,p32,p33,p34,p35,p36,p37,p38,p39,p40,p41,p42,p43,p44,p45,p46,p47,p48,p49,p50,p51,p52,
                        p53,p54,p55,p56,p57,p58,p59,p60,p61,p62,p63,p64,p65,p66,p67,p68,p69,p70,p71])




string_matches = string_matcher(data)

# print(string_matches)

d1=[]
for match_id, start, end in string_matches:
    rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
    span = data[start : end]  # get the matched slice of the doc
    d1.append(span.text)
    keyterm = span.text
# print(d1)
total_terms = Counter(d1)
rank_term = total_terms.most_common() #list all
# total_terms.most_common()[:200] #top 200
# rank_term
all_corpora = data.text
silver_corpus = d1

tokens = word_tokenize(all_corpora)
offset = 0
entities = []
i = 0
while i in  range(len(tokens)):
    offset = all_corpora[0].find(tokens[i], offset)
    if i < len(tokens) - 3 and " ".join(tokens[i:i+3]) in silver_corpus:
        entities.append((offset,offset+len(tokens[i]),'B'))
        entities.append((offset+len(tokens[i])+1,offset+len(tokens[i])+len(tokens[i+1])+1,'I'))
        entities.append((offset+len(tokens[i])+len(tokens[i+1])+1,offset+len(tokens[i])+len(tokens[i+1])+len(tokens[i+2])+2,'I'))
        #offset = offset+len(tokens[i])+len(tokens[i+1])
        i = i+3
    elif i < len(tokens) - 2 and " ".join(tokens[i:i+2]) in silver_corpus:
        entities.append((offset,offset+len(tokens[i]),'B'))
        entities.append((offset+len(tokens[i])+1,offset+len(tokens[i])+len(tokens[i+1])+1,'I'))
        i = i+2
    elif i < len(tokens) - 1 and " ".join(tokens[i:i+1]) in silver_corpus:
        entities.append((offset,offset+len(tokens[i]),'B'))
        i = i+1
    else:
        entities.append((offset,offset+len(tokens[i]),'O'))
        i = i+1
print(entities)

nlp = spacy.blank("en")
doc_bin = DocBin(attrs=["ENT_IOB"])
doc_bin.add(entities)
doc_bin.to_disk("./train.spacy")