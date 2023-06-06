import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

# Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the trained model
model.load_state_dict(torch.load('path/to/saved/model'))

# Evaluate on the test data
test_texts = ['Test sentence 1', 'Test sentence 2', ...]
test_labels = [0, 1, ...]  # Assuming binary classification with 0 and 1 as labels

# Tokenize the test texts
encoded_test_inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')

# Create a TensorDataset for the test data
test_dataset = TensorDataset(encoded_test_inputs['input_ids'], encoded_test_inputs['attention_mask'], torch.tensor(test_labels))

# Create a DataLoader for the test data
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set the model in evaluation mode
model.eval()

# Variables to track accuracy and total examples
total_examples = 0
correct_predictions = 0

# Iterate over the test data
for batch in test_dataloader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    targets = batch[2].to(device)

    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Get predicted labels
        _, predicted_labels = torch.max(logits, dim=1)

        # Count correct predictions
        correct_predictions += (predicted_labels == targets).sum().item()

    total_examples += targets.size(0)

# Calculate accuracy
accuracy = correct_predictions / total_examples
print(f'Test Accuracy: {accuracy}')
