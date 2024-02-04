from transformers import AutoModel, AutoTokenizer

# Specify the model name
model_name = 'bert-base-uncased'

# Load the model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to a local directory
model.save_pretrained('local_model')
tokenizer.save_pretrained('local_model')
