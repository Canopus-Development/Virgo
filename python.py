# python.py

from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_python_code():
    model_name = "salesforce/codegen-6b-mono"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add logic to get user input or use a predefined prompt for Python code generation
    user_input = input("Enter Python code or describe the task: ")

    # Tokenize input text
    input_text = [user_input]
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate output
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        num_return_sequences=1,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        early_stopping=True,
        no_repeat_ngram_size=2,
        do_sample=True
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Display the generated Python code
    print("Generated Python code:")
    print(decoded_output)
