import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel

# 讀取CSV檔案
df = pd.read_csv('items_colors.csv')

# 將物品和顏色轉換為BERT的輸入
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
item_tokens = tokenizer(df['item'].tolist(), padding=True, truncation=True, max_length=16, return_tensors='pt')
color_tokens = tokenizer(df['color'].tolist(), padding=True, truncation=True, max_length=8, return_tensors='pt')

# 將BERT輸入轉換為BERT輸出
bert_model = BertModel.from_pretrained('bert-base-uncased')
item_outputs = bert_model(**item_tokens)[0]
color_outputs = bert_model(**color_tokens)[0]

# 將BERT輸出作為LSTM輸入
class ColorPredictor(nn.Module):
    def __init__(self):
        super(ColorPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=128, out_features=1)
        
    def forward(self, item_outputs, color_outputs):
        inputs = torch.cat([item_outputs, color_outputs], dim=1)
        _, (hn, cn) = self.lstm(inputs)
        outputs = self.fc(hn[-1])
        return outputs

# 訓練模型
model = ColorPredictor()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for i in range(len(df)):
        item_output = item_outputs[i:i+1]
        color_output = color_outputs[i:i+1]
        label = torch.tensor([int(df['item'][i] == df['color'][i])], dtype=torch.float32)

        optimizer.zero_grad()
        output = model(item_output, color_output)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(df)))

# 測試模型
test_items = ['apple', 'banana', 'orange', 'grape', 'lemon']
test_colors = ['red', 'yellow', 'orange', 'purple', 'green']

for i in range(len(test_items)):
    test_item_tokens = tokenizer(test_items[i], padding=True, truncation=True, max_length=16, return_tensors='pt')
    test_color_tokens = tokenizer(test_colors[i], padding=True, truncation=True, max_length=8, return_tensors='pt')
    test_item_output = bert_model(**test_item_tokens)[0]
    test_color_output = bert_model(**test_color_tokens)[0]
    test_output = model(test_item_output, test_color_output)
    test_prediction = torch.sigmoid(test_output) > 0.5
    print('%s: %s (%s)' % (test_items[i], test_colors[i], 'correct' if test_prediction.item() else 'incorrect'))
