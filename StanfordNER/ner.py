from html import entities
from lib2to3.pgen2 import token
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordNERTagger
import csv
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer


"""
This script only tests the model
"""


train_split = 0.80

PATH_TO_JAR='stanford-ner-2020-11-17/stanford-ner.jar'
PATH_TO_MODEL = 'cooking-ner-model_4.5.ser.gz'

tagger = StanfordNERTagger(model_filename=PATH_TO_MODEL,path_to_jar=PATH_TO_JAR, encoding='utf-8')

df = pd.read_csv('../data/annotations_combined_v2.csv')

##creation of training data
Instructions = list(df['instructions'])
Verbs = list(df['Verbs'])
data = []
print("Length of instructions:",len(Instructions))
for i in range(0,len(Instructions)):
    verb_ = Verbs[i]
    verbs = []
    if not isinstance(verb_, str): # To catch the Nan values
        verb_ = 'none'
    verb_ = verb_.split(',')
    for v in verb_:
        temp = word_tokenize(v)
        verbs = verbs + temp
    instruction = Instructions[i]
    # Lower case everything
    if not isinstance(instruction, str):
        instruction = "none"
    else:
        instruction = instruction.lower()
    tokens = word_tokenize(instruction)
    cooking_verbs = [v.lower() for v in verbs]
    inst_tokens = [t.lower() for t in tokens]

    # Label the tokens as PROC or as O. PROC - cooking actions, O - other
    tags = []
    for tok in inst_tokens:
        if tok in cooking_verbs:
            tags.append('PROC')
        else:
            tags.append('O')
    data.append((tokens, tags))

print("Total length of the data:",len(data))
split = int(len(data) * train_split)
test_data = data[split:]
training_data = data[:split]
print("Length of training data:", len(training_data))
print("Length of testing data:", len(test_data))

##writing the training data in the required
out_file = open('training_data.tsv', 'wt')
print("Writing training data to tsv...")
tsv_writer = csv.writer(out_file, delimiter='\t')
for data in training_data:
    tokens = data[0]
    tags = data[1]
    for i in range(len(tokens)):
        tsv_writer.writerow([tokens[i],tags[i]])
    tsv_writer.writerow(" ")
out_file = open('test_data.tsv', 'wt')
tsv_writer = csv.writer(out_file, delimiter='\t')
for data in test_data:
    tokens = data[0]
    tags = data[1]
    for i in range(len(tokens)):
        tsv_writer.writerow([tokens[i],tags[i]])
    tsv_writer.writerow(" ")


# df = pd.read_csv('/home/anirudh/aiisc/stanfordNER/stanford-ner-2020-11-17/train/test_data.tsv',sep = '\t')
##testing the trained model
print("Testing the trained model...")
confusion_matrix = {}
confusion_matrix['true-positive'] = 0
confusion_matrix['false-positive'] = 0
confusion_matrix['true-negative'] = 0
confusion_matrix['false-negative'] = 0
test_size = len(test_data)
matches = 0
mismatches = 0
for d in test_data:
    inst_tokens = d[0]
    ground_truth = d[1]
    # Test the model. Returns list of tuples
    tagged = tagger.tag(inst_tokens)
    predicted_tags = []
    entities_ = []
    for tags in tagged:
        entities_.append(tags[0])
        predicted_tags.append(tags[1])
    
    for i in range(len(ground_truth)):

        if predicted_tags[i]!=ground_truth[i]:
            # Predicted wrong for both PROC and O
            if predicted_tags[i] == 'PROC': # Falsely predicted O as PROC
                confusion_matrix['false-positive']+=1
            elif predicted_tags[i] == 'O': 
                confusion_matrix['false-negative']+=1
        else:
            # Predicted right for both PROC and O
            if predicted_tags[i] == 'PROC': 
                confusion_matrix['true-positive']+=1
            elif predicted_tags[i] == 'O': 
                confusion_matrix['true-negative']+=1

print("Confusion Matrix :",confusion_matrix)

