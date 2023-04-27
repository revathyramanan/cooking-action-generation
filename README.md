# Cook-Gen
This repository contains implementation of methods experimented and introduced in the paper titled "Cook-Gen: Robust Generative Modeling of Cooking Actions from Recipes". 

### Rule Based Approaches
* A dictionary look-up table
* Code available in the folder Rule-based
#### Industry Standards
* Spacy POS Tagger
* Stanford NER
* Code available under respective folders
#### Large Language Models
* ELECTRA
* XL-Net
* Code is available inside the folder Electra-XLNet.

### Generative Models
The proposed CookGen model can be found under the folder CookGen. It contains both the CookGen-NN and CookGen-PF.

### ChatGPT
We used ChatGPT model `gpt-3.5-turbo` to generate cooking actions given cooking instructions. The model is accessed through the API exposed by OpenAI. The promt used is shown in the figure below
![ChatGPT prompt](Diagrams/chatgpt_prompt.pdf)

### Additional Resources
