**Instructions**

Custom NER model

Installation, training and testing instructions given here: https://medium.com/@subshakya591/custom-trained-nltk-stanford-ner-tagger-and-spacy-ner-tagger-comparison-d4e20e41b0bf
Download stanford-ner zip from (download section) - https://nlp.stanford.edu/software/CRF-NER.shtml

Our use case is to identify the cooking verb instructions such as bake,saute etc, which have been labelled as 'PROC'.

The pretrained model is available under the name ''.

Command to train the model
cd stanford-ner-tagger/
java -cp "stanford-ner.jar:lib/*" -mx4g edu.stanford.nlp.ie.crf.CRFClassifier -prop train_prop.txt

