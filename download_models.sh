#!/bin/bash

# Download models and tokens required for Virgo

# Download Codegen model
wget https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/pytorch_model.bin -P models/codegen/
wget https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/tokenizer.json -P models/codegen/
wget https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/config.json -P models/codegen/

# Download Error Detection model
wget https://huggingface.co/codellama/CodeLlama-7b-hf/resolve/main/pytorch_model.bin -P models/error_detection/
wget https://huggingface.co/codellama/CodeLlama-7b-hf/resolve/main/tokenizer.json -P models/error_detection/
wget https://huggingface.co/codellama/CodeLlama-7b-hf/resolve/main/config.json -P models/error_detection/

# Download ConvNeXt V2 model for General Coding Assistant
wget https://huggingface.co/model_url/resolve/main/pytorch_model.bin -P models/general_coding_assistant/
wget https://huggingface.co/model_url/resolve/main/tokenizer.json -P models/general_coding_assistant/
wget https://huggingface.co/model_url/resolve/main/config.json -P models/general_coding_assistant/
