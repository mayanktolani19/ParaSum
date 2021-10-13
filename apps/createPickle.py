from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle

model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(model, open('tokenizer.pkl', 'wb'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)