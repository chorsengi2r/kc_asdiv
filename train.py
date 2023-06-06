import torch
import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

# Prepare the training data
print("Preparing training data...")
#texts = ['Example sentence 1', 'Example sentence 2', ...]
#labels = [0, 1, ...]  # Assuming binary classification with 0 and 1 as labels
# Opening JSON file
f = open('asdiv_train_test.json')

# returns JSON object as 
# a dictionary
data = json.load(f)
train_data = data['train']
test_data = data['test']

train_texts = []
train_labels = []
for question in train_data:
    train_texts.append(question['text'])
    train_labels.append(question['label'])
print("Text:", len(train_texts), "Labels:", len(train_labels))
num_labels = pd.Series(train_labels).nunique()
print("Unique training labels:", num_labels)
print(pd.Series(train_labels).unique())

# Convert string labels to numerical labels
label_map = {label: i for i, label in enumerate(set(train_labels))}
numerical_labels = [label_map[label] for label in train_labels]

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameters
epochs = 10
batch_size = 16
learning_rate = 2e-5

# Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Move the model to the device
model.to(device)


# Tokenize the input texts
encoded_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')

# Move the encoded inputs and labels to the device
input_ids = encoded_inputs['input_ids'].to(device)
attention_mask = encoded_inputs['attention_mask'].to(device)
labels = torch.tensor(numerical_labels).to(device)

# Create a TensorDataset
#dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], torch.tensor(train_labels))
dataset = TensorDataset(input_ids, attention_mask, labels)


# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    model.train()

    for batch in dataloader:
        input_ids = batch[0]
        attention_mask = batch[1]
        targets = batch[2]

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Calculate loss
        loss = loss_fn(logits, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{epochs} - Loss: {average_loss}')

# Save the trained model
model.save_pretrained('./kc_model/')
tokenizer.save_pretrained('./kc_tokenizer')
