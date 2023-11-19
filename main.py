import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel


def generate_code(input_text):
    try:
        model_name = "Salesforce/codegen-350M-mono"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Tokenize input text
        input_text = [input_text]
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate output
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=1000, num_return_sequences=1)

        # Decode the generated output
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        return decoded_output
    except Exception as e:
        return f"An error occurred: {str(e)}"

def show_readme():
    if not os.path.exists("README.md"):
        print("Readme file not found.")
    else:
        with open("README.md", "r") as readme_file:
            readme_content = readme_file.read()
            print(readme_content)

def developer_assistant(task, input_text, input_file=None, output_file=None):
    if task == "generate code":
        generated_code = generate_code(input_text)

        with open("output_code.txt", "w") as file:
            file.write(generated_code)

        return "Generated code saved to output_code.txt file"
    else:
        return "Task not supported. Please specify a valid task for the developer assistant."


def user_input_prompt():
    show_readme()
    task = input("Enter the task ('generate code', 'general'): ")

    if task.lower() == "general":
        chatbot()
    else:
        input_text = input("Enter the input text: ")

        if task in ["generate code"]:
            print(developer_assistant(task, input_text))
        elif task == "general":
            print(chatbot(input_text))
        else:
            print("Invalid task.")


def chatbot():
    # Load pre-trained GPT-3 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Chatbot loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Tokenize user input and generate bot response
        input_ids = tokenizer.encode(user_input, return_tensors='pt')
        bot_response = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
        bot_reply = tokenizer.decode(bot_response[0], skip_special_tokens=True)
        print(f"Chatbot: {bot_reply}")


def main():
    user_input_prompt()

if __name__ == "__main__":
    main()
