from transformers import pipeline
summarizer = pipeline('summarization')
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_response(context,return_sequences):
  text = "paraphrase: "+context + " </s>"
  encoding = tokenizer.encode_plus(text, max_length=128, padding=True, return_tensors="pt")
  input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
  model.eval()
  diverse_beam_outputs = model.generate(
      input_ids=input_ids,attention_mask=attention_mask,
      early_stopping=True,
      num_beams=5,
      num_beam_groups = 5,
      num_return_sequences=return_sequences,
      diversity_penalty = 0.70
  )
  for beam_output in diverse_beam_outputs:
      sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
  return sent


def text_paraphrase(input):

  from sentence_splitter import SentenceSplitter, split_text_into_sentences

  splitter = SentenceSplitter(language='en')

  sentence_list = splitter.split(input)

  # Do a for loop to iterate through the list of sentences and paraphrase each sentence in the iteration
  paraphrase = []

  for i in sentence_list:
    a = get_response(i,1)
    a = a[18:]
    paraphrase.append(a)

  paraphrase2 = [''.join(x) for x in paraphrase]
  # paraphrase2

  # Combines the above list into a paragraph
  paraphrase3 = [''.join(x for x in paraphrase2) ]
  paraphrased_text = str(paraphrase3).strip('[]').strip("'")
  # paraphrased_text
  return paraphrased_text


def text_summarize(ARTICLE, maxLength, minLength):
  output = summarizer(ARTICLE)[0]['summary_text']
  ans = text_paraphrase(output)
  return ans
