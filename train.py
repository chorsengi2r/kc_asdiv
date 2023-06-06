import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

# Define the hyperparameters
epochs = 5
batch_size = 32
learning_rate = 2e-5

# Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Prepare the training data
texts = ['Example sentence 1', 'Example sentence 2', ...]
labels = [0, 1, ...]  # Assuming binary classification with 0 and 1 as labels

# Tokenize the input texts
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Create a TensorDataset
dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], torch.tensor(labels))

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
model.save_pretrained('path/to/save/model')
tokenizer.save_pretrained('path/to/save/tokenizer')