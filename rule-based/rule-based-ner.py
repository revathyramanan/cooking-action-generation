import json
import pandas as pd
import spacy
import nltk
from nltk.tokenize import word_tokenize
import random
# from nltk.tag.stanford import StanfordNERTagger

train_split = 0.80
random.seed(1234)

def create_data():
    df = pd.read_csv('../data/annotations_combined_v2.csv')
    ##creation of training data
    Instructions = list(df['instructions'])
    Verbs = list(df['Verbs'])
    data = []
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

        data.append((inst_tokens, cooking_verbs))

    split = int(len(data) * train_split)
    test_data = data[split:]
    training_data = data[:split]
    return training_data, test_data


def get_cooking_vocab(training_data):
    vocab = []
    for inst, verbs in training_data:
        vocab = vocab + verbs
    cooking_verbs = list(set(vocab))
    try:
        cooking_verbs.remove('.')
        cooking_verbs.remove('a')
        cooking_verbs.remove('or')
        cooking_verbs.remove('of')
        cooking_verbs.remove('the')
        cooking_verbs.remove('is')
        cooking_verbs.remove('in')
        cooking_verbs.remove('to')
        cooking_verbs.remove('and')
        cooking_verbs.remove(",")
    except:
        pass
    return cooking_verbs

def test(test_data, cooking_vocab):
    TP = []
    FP = []
    FN = []
    TN = []
    for inst, truth_verbs in test_data:
        # PROC predcited as PROC
        predicted_PROC = set(cooking_vocab).intersection(set(inst))
        pred_truth_intersec = set(predicted_PROC).intersection(set(truth_verbs))
        TP.append(len(pred_truth_intersec))

        # Tokens falsely predicted as PROC (Other tokens predicted as PROC)
        oth_pred_as_PROC1 = set(predicted_PROC).difference(set(pred_truth_intersec))
        FP.append(len(oth_pred_as_PROC1))

        # PROC missed (PROC falsely predicted as Other tokens)
        PROC_pred_as_oth = set(truth_verbs).difference(set(pred_truth_intersec))
        FN.append(len(PROC_pred_as_oth))

        # Other tokens predicted as "other"
        other = set(inst).difference(set(truth_verbs))
        oth_pred_as_PROC = set(other).intersection(predicted_PROC)
        predicted_oth = set(other).difference(set(oth_pred_as_PROC))
        TN.append(len(predicted_oth))


    return sum(TP), sum(FP), sum(FN), sum(TN)


def get_data_spacy():
    df = pd.read_csv('annotations_combined_v2.csv')
    ##creation of training data
    Instructions = list(df['instructions'])
    Verbs = list(df['Verbs'])
    data = []
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
        cooking_verbs = [v.lower() for v in verbs]

        data.append((instruction, cooking_verbs))

    split = int(len(data) * train_split)
    test_data = data[split:]
    training_data = data[:split]
    return training_data, test_data



def test_with_spacy(test_data, cooking_vocab):
    nlp = spacy.load("en_core_web_lg")
    TP = []
    FP = []
    FN = []
    TN = []
    for inst_, truth_verbs in test_data:

        doc = nlp(inst_)
        spacy_verbs = []
        inst_tokens = word_tokenize(inst_)
        inst = [t.lower() for t in inst_tokens]
        for ent in doc:
            if str(ent.pos_) == 'VERB':
                spacy_verbs.append(ent.text)
        
        # PROC predcited as PROC
        predicted_PROC = set(cooking_vocab).intersection(set(spacy_verbs))
        pred_truth_intersec = set(predicted_PROC).intersection(set(truth_verbs))
        TP.append(len(pred_truth_intersec))

        # Tokens falsely predicted as PROC (Other tokens predicted as PROC)
        oth_pred_as_PROC1 = set(predicted_PROC).difference(set(pred_truth_intersec))
        FP.append(len(oth_pred_as_PROC1))

        # PROC missed (PROC falsely predicted as Other tokens)
        PROC_pred_as_oth = set(truth_verbs).difference(set(pred_truth_intersec))
        FN.append(len(PROC_pred_as_oth))

        # Other tokens predicted as "other"
        other = set(inst).difference(set(truth_verbs))
        oth_pred_as_PROC = set(other).intersection(predicted_PROC)
        predicted_oth = set(other).difference(set(oth_pred_as_PROC))
        TN.append(len(predicted_oth))


    return sum(TP), sum(FP), sum(FN), sum(TN)






def main():
    train_data, test_data = create_data()
    cooking_vocab = get_cooking_vocab(train_data)
    tp, fp, fn, tn = test(test_data, cooking_vocab)
    print("Length of training data", len(train_data))
    print("Length of Testing data:", len(test_data))
    print("True Positive:", tp)
    print("False Positive:", fp)
    print("False Negative:", fn)
    print("True Negative:", tn)

    print("\n ----------- SPACY")
    train_data, test_data = get_data_spacy()
    tp, fp, fn, tn = test_with_spacy(test_data, cooking_vocab)
    print("True Positive:", tp)
    print("False Positive:", fp)
    print("False Negative:", fn)
    print("True Negative:", tn)


    
    # json.dump(new_data, open('../data/test_data_cooking.json', 'w'))
        

main()