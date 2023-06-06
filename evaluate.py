import torch
import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare the test data
print("Preparing test data...")
# Opening JSON file
f = open('asdiv_train_test.json')

# returns JSON object as 
# a dictionary
data = json.load(f)
test_data = data['test']

test_texts = []
test_labels = []
for question in test_data:
    test_texts.append(question['text'])
    test_labels.append(question['label'])
print("Text:", len(test_texts), "Labels:", len(test_labels))
num_labels = pd.Series(test_labels).nunique()
print("Unique test labels:", num_labels)

# Convert string labels to numerical labels
label_map = {label: i for i, label in enumerate(set(test_labels))}
numerical_labels = [label_map[label] for label in test_labels]


# Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Move the model to the device
model.to(device)

# Load the trained model
model.load_state_dict(torch.load('./kc_model/pytorch_model.bin'))

# Tokenize the test texts
encoded_test_inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')

# Move the encoded test inputs and labels to the device
input_ids = encoded_test_inputs['input_ids'].to(device)
attention_mask = encoded_test_inputs['attention_mask'].to(device)
labels = torch.tensor(numerical_labels).to(device)

# Create a TensorDataset for the test data
test_dataset = TensorDataset(input_ids, attention_mask, labels)

# Create a DataLoader for the test data
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set the model in evaluation mode
model.eval()

# Set batch size
batch_size = 32

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
