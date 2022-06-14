# Terminology
This README file describes the steps and the usage of the code for "Develop a Term Identification System fora Specific Domain" project.

--------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------------------------------------

Step 1: Read all PDF files and convert to text format

def convertpdfstotext(path) function is defined to read all the PDF file in a folder and is defined with textract function used for extracting contents from PDF file (converting PDF to text) in Python.

Note: path = r'C:\path\to\folder' 


--------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------------------------------------

Step 2: Text processing

def preprocessing(convertpdftotext) function is definded for text processing. This includes some main pre-processing tasks, such as stopword removal, tokenization, lemmatization, punctuation removal, URL removal, and more.



--------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------------------------------------

Step 3: Define rule-based system for phrase pattern matching

Spacy is used for this task.

matches = matcher(data)

Note: data is derived from data = nlp(cleandata) where convert the data derived from the processing part above to Spacy format.


--------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------------------------------------

Step 4: Define IOB tagging rule-based system for annotating all corpora

def iob_rule(all_corpora,silver_corpus) function is defiend for annotating all corpora with IOB format. It gets all corpora and list of terms as input, and by matching between the list of terms and all corpora, it will produce tokens and entities (IOB).

Note: all_corpora refers to all the PDF files
      silver_corpus refers to a list of terms derived from the previous steps

"Terminology Term Extraction.ipynb" contains code to generate above steps

--------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------------------------------------

Model Training

Step 5: "Final_corpus" file generates train.spacy file ( annotated corpus with iob tags)
        "unseen_article" file generates dev.spacy file (preprocessed unseen article from different domain)

Comands used for training and evaluations:

python -m spacy train config.cfg –output ./Report –paths.train ./train –paths.dev ./dev

python -m spacy evaluate model data path –output –code –gold-preproc –gpu-id –displacy-path –


Report folder has "model-best" and "model-last" folders which contains all data generated during training 
                   "metrics.json" file gives the token_p,token_r,token_f scores generated after running evaluation command
                   




   
-------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#-----------
------------------------------

(i) your initial textual corpus :
Path : Terminology Final Project_Shalini_Soklay ->initial textual corpus

(ii) the lexicon of terms (both versions before and after manual filtering)
Path :Terminology Final Project_Shalini_Soklay ->first_time_keyterm_extraction_before manual.xlsx
Path :Terminology Final Project_Shalini_Soklay ->string_matching_keyterm_extraction.xlsx

(iii)the annotated training dataset 
Path :Terminology Final Project_Shalini_Soklay ->annotated training result.xlsx

(iv) annotation results on test text with evaluation
Path :Terminology Final Project_Shalini_Soklay ->Extracted_keywords_after_training.xlsx
