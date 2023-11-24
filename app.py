# app.py

from python import generate_python_code
from multi import generate_multi_language_code

def main():
    print("Welcome to Virgo Developer Assistant!")
    print("Select an option:")
    print("1. Generate Python Code")
    print("2. Generate Code in Multiple Languages")
    print("0. Exit")

    while True:
        choice = input("Enter your choice (0-2): ")

        if choice == "1":
            generate_python_code()
        elif choice == "2":
            generate_multi_language_code()
        elif choice == "0":
            print("Exiting Virgo Developer Assistant...")
            break
        else:
            print("Invalid choice. Please select again.")

if __name__ == "__main__":
    main()
