# multi.py

from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_code_for_language(language):
    model_name = "salesforce/codegen-6b-multi"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Get user input based on the selected language
    if language == "javascript":
        user_input = input("Enter JavaScript code or describe the task: ")
    elif language == "java":
        user_input = input("Enter Java code or describe the task: ")
    elif language == "c":
        user_input = input("Enter C code or describe the task: ")
    elif language == "cpp":
        user_input = input("Enter C++ code or describe the task: ")
    elif language == "csharp":
        user_input = input("Enter C# code or describe the task: ")
    elif language == "ruby":
        user_input = input("Enter Ruby code or describe the task: ")
    elif language == "go":
        user_input = input("Enter Go code or describe the task: ")
    else:
        user_input = input(f"Enter {language.capitalize()} code or describe the task: ")

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

    # Display the generated code for the selected language
    print(f"Generated {language.capitalize()} code:")
    print(decoded_output)

def generate_multi_language_code():
    print("Select the language for code generation:")
    print("1. JavaScript")
    print("2. Java")
    print("3. C")
    print("4. C++")
    print("5. C#")
    print("6. Ruby")
    print("7. Go")
    print("0. Exit")

    while True:
        choice = input("Enter your choice (0-7): ")

        if choice == "1":
            generate_code_for_language("javascript")
        elif choice == "2":
            generate_code_for_language("java")
        elif choice == "3":
            generate_code_for_language("c")
        elif choice == "4":
            generate_code_for_language("cpp")
        elif choice == "5":
            generate_code_for_language("csharp")
        elif choice == "6":
            generate_code_for_language("ruby")
        elif choice == "7":
            generate_code_for_language("go")
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select again.")
