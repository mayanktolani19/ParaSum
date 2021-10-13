from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle
# import warnings
# warnings.filterwarnings('ignore')

# model = pickle.load(open('./apps/model.pkl', 'rb'))
# tokenizer = pickle.load(open('./apps/tokenizer.pkl', 'rb'))
model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
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
      max_length=128,
      num_beams=5,
      num_beam_groups = 5,
      num_return_sequences=return_sequences,
      diversity_penalty = 0.70
  )
  for beam_output in diverse_beam_outputs:
      sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
  return sent


def text_paraphrase(input):
  # input = "The increase in demand for information security has made cryptography a basic need for protecting data from unauthorized access. In this paper, we have presented a computationally inexpensive Enyo block cipher. This multi-layered algorithm provides a promising solution to safeguard user's information. The cipher works on the same line as the Feistel-based network with the symmetric structure and operates on 8 bit ASCII characters. Using 6-bit custom encoding and a secret key generates 6-bit encrypted text. This cipher modeled using undemanding operations utilizes partitioned key-based encryption with unique mapping followed by various bitwise swaps, a shifted modulo encryption, and uses a transposition matrix. The proposed cipher is highly customizable as per user demands and necessities, that make it more dependent on user inputs. Enyo cipher is faster in encryption and decryption than the Playfair cipher and shows comparable performance with XOR encryption. Enyo cipher demonstrates good resistance to a brute-force attack. It is well suited for small scale applications where the computational power is a bottleneck. A comparison is also made that shows the impact of the proposed cipher to commonly available classical ciphers." #@param {type:"string"}
  # Takes the input paragraph and splits it into a list of sentences

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

  # Comparison of the original (input variable) and the paraphrased version (paraphrase3 variable)
  print("Output Text:\n")
  print(paraphrased_text)
  return paraphrased_text