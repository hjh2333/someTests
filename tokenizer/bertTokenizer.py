from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

text = '[CLS] 武1松1打11老虎 [SEP] 你在哪 [SEP]frwfg.fesi'
tokenized_text = tokenizer.tokenize(text)#切词 方式1
print(tokenized_text)
# token_samples_a = tokenizer.convert_tokens_to_ids(tokenized_text)#只返回token_ids,手动添加CLS与SEP

# token_samples_b=tokenizer(text)#返回一个字典，包含id,type,mask，无须手动添加CLS与SEP 方式2

# token_samples_c=tokenizer.encode(text=text)#只返回token_ids，无须手动添加CLS与SEP 方式3

# token_samples_d=tokenizer.encode_plus(text=text,max_length=30,return_tensors='pt')#方式4 返回一个字典，包含id,type,mask，无须手动添加CLS与SEP，可以指定返回类型与长度