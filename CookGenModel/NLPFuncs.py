import spacy
from Dataloader import Dataloader
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

Dataloader.prepare_data()

class NLPFuncs(object):

    model = spacy.load('en_core_web_sm')
    embedding_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    @staticmethod
    def embed_input(text):

        lower_case = text.lower()
        tokens = torch.tensor([NLPFuncs.tokenizer.encode(lower_case)])
        return torch.mean(torch.squeeze(NLPFuncs.embedding_model(tokens)[0],0).t(),dim=-1)
        

    @staticmethod
    def infer_verbs(text,model):

        doc = NLPFuncs.model(text.lower())
        data_embeddings = [NLPFuncs.embed_input(other_text[0]) for other_text in Dataloader.text_data]
        text_embedding = NLPFuncs.embed_input(text)
        distances = [torch.sum(torch.pow(text_embedding-other_embedding,2)) for other_embedding in tqdm(data_embeddings)]
        min_distance = min(distances)
        min_index = distances.index(min_distance)
        verbs = [token.text for token in doc if token.pos_ == "VERB"]
        return min_index, verbs
