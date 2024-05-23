import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.cuda as cuda
dir="Helsinki-NLP/opus-mt-zh-en"
tokenizer_zh2en = AutoTokenizer.from_pretrained(dir)
model_zh2en = AutoModelForSeq2SeqLM.from_pretrained(dir)

def zh2en(text):
    print("翻译",text)
    try:
        tokenized_text = tokenizer_zh2en([text], return_tensors='pt')
        translation = model_zh2en.generate(**tokenized_text, max_new_tokens=1024)
        cuda.empty_cache()
        con= tokenizer_zh2en.batch_decode(translation, skip_special_tokens=True)[0]
        print("结果",con)
    except:
        con=text
    return con



 