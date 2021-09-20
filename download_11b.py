from transformers import AutoTokenizer, AutoModelWithLMHead
  
tokenizer = AutoTokenizer.from_pretrained("t5-11b", cache_dir="/workspace/cache")
model = AutoModelWithLMHead.from_pretrained("t5-11b", cache_dir="/workspace/cache")
