import json
import pandas as pd
import spacy
import nltk
from nltk.tokenize import word_tokenize
import random
import requests
import json
from dotenv import load_dotenv
import os
import openai
import time
load_dotenv()

# Replace it with your own API Key
openai.api_key = os.getenv("API_KEY")

train_split = 0.80
random.seed(1234)

def get_data():
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
        cooking_verbs = [v.lower() for v in verbs]

        data.append((instruction, cooking_verbs))

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

def tokenize(res):
    res = res.split(",")
    predicted_tokens = [t.lower() for t in res]
    return predicted_tokens


def testGPT(testing_data):
    TP = []
    FP = []
    FN = []
    TN = []
    predicted_tokens = []
    insts = []
    truth_verbs = []
    counter = 0
    for inst, truth_verb in testing_data[1573:]:
        counter = counter + 1
        prompt = 'get cooking verbs from:' + ' ' + str(inst) + '.' + 'If no cooking verbs, return only none. Return cooking verbs with no special characters.'
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
        )
        try:
            result = completion.choices[0].message.content
            predicted = tokenize(result)
        except:
            predicted = []
        
        print(inst, result)
        print("Counter:", counter)

        insts.append(inst)
        truth_verbs.append(truth_verb)
        predicted_tokens.append(predicted)
        if counter == 24:
            pass
        if counter % 10 == 0:
            data_dict = {'inst': insts, 'truth_verbs': truth_verbs, 'predicted': predicted_tokens}
            json.dump(data_dict, open('chatgpt_results_new.json', 'w'))
 
    data_dict = {'inst': insts, 'truth_verbs': truth_verbs, 'predicted': predicted_tokens}
    json.dump(data_dict, open('chatgpt_results_new.json', 'w'))
        


def result_analysis(cooking_vocab):
    dict1 = json.load(open('chatgpt_results.json', 'r'))

    instructions = dict1['inst']
    truth = dict1['truth_verbs']
    predicted_tokens = dict1['predicted']

    TP = []
    FP = []
    TN = []
    FN = []

    for idx, inst_ in enumerate(instructions):
        inst_ = word_tokenize(inst_)
        inst = [i.lower() for i in inst_]

        truth_verbs = truth[idx]

        predicted = predicted_tokens[idx]
        word = " ".join(predicted)
        predicted = word_tokenize(word)
        # PROC predcited as PROC
        predicted_PROC = set(cooking_vocab).intersection(set(predicted))
        correctly_predicted_PROC = set(predicted_PROC).intersection(set(truth_verbs))
        TP.append(len(correctly_predicted_PROC))

        # PROC missed (PROC falsely predicted as Other tokens)
        PROC_pred_as_oth = set(truth_verbs).difference(set(correctly_predicted_PROC))
        FN.append(len(PROC_pred_as_oth))

        
        other = set(inst).difference(set(truth_verbs))
        oth_pred_as_PROC = set(other).intersection(predicted_PROC)
        FP.append(len(oth_pred_as_PROC))

        # Other tokens predicted as "other"
        predicted_oth = set(other).difference(set(oth_pred_as_PROC))
        TN.append(len(predicted_oth))
    

    
    return sum(TP), sum(FP), sum(FN), sum(TN)



def main():
    training_data, testing_data = get_data()
    cooking_vocab = get_cooking_vocab(training_data)
    testGPT(testing_data)
    tp, fp, fn, tn = result_analysis(cooking_vocab)
    print(tp, fp, fn, tn)



main()