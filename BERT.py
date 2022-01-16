import torch
import os
import pandas as pd
from transformers import BertTokenizer, BertModel

if not os.path.exists('processed'):
    os.mkdir('processed')

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embeddingExtract(data):
    raw = pd.read_csv('./data/{}.csv'.format(data))
    df = pd.concat([raw.iloc[:, 1], raw.iloc[:, 2]], ignore_index=True)

    num_data = len(df)
    print('Total Idx = ', num_data)

    feature = torch.zeros((num_data, 768), device=cuda)
    for i, comment in enumerate(df):
        if int(i) % 1000 == 0:
            print('{} | {}/{}'.format(data, i, num_data))

        inputs = tokenizer(comment, return_tensors="pt", truncation=True).to(cuda)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = torch.squeeze(torch.mean(last_hidden_states, dim=1))
        feature[i] = embedding

    return feature

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(cuda)

DATA = 'validation_data'
print('\nExtracting Embedding | {}.csv\n'.format(DATA))
SAVE_PATH = './processed/BERT_{}.pt'.format(DATA)
output = embeddingExtract(DATA).cpu()

torch.save(output, SAVE_PATH)
print('\nSaved BERT_{}.pt'.format(DATA))
print('Done')
