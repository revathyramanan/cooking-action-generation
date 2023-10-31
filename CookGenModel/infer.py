from CookGen import CookGen
from NLPFuncs import NLPFuncs
from Dataloader import Dataloader
import torch
import gc

class CookGenModelWrapper(object):

    def __init__(self):

        self.model = CookGen()
        self.model.load_state_dict(torch.load("modelCookGenClass.pt"))

    def infer(self,text):

        index, verbs = NLPFuncs.infer_verbs(text,self.model)
        model_logits = self.model([index])
        model_prediction = torch.argmax(model_logits)
        predicted_label = Dataloader.label_data[model_prediction]
        return [predicted_label] + verbs