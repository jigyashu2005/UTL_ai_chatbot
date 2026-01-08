import os
import logging
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, AuthenticationError
from my_logger import setup_logging
from connectivity import check_connectivity

def get_chat_completion(client, prompt, model="gpt-3.5-turbo"):
    """
    Sends a prompt to the OpenAI API and retrieves the response.

    Args:
        client (OpenAI): The OpenAI client instance.
        prompt (str): The user prompt.
        model (str): The model to use.

    Returns:
        str: The content of the response, or None if an error occurred.
    """
    try:
        logging.info(f"Sending prompt to OpenAI: {prompt}")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        logging.info("Received response from OpenAI.")
        return content
        
    except RateLimitError as e:
        logging.error(f"Rate limit exceeded: {e}")
        print("Error: You have hit the rate limit. Please try again later.")
    except AuthenticationError as e:
        logging.error(f"Authentication failed: {e}")
        print("Error: Authentication failed. Please check your API key.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"Error: An unexpected error occurred. Check app.log for details.")
    
    return None

def main():
    """
    Main function to execute the script.
    """
    setup_logging()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables.")
        print("Error: OPENAI_API_KEY not set. Please create a .env file.")
        return

    # Check connectivity
    if not check_connectivity():
        print("Error: No internet connection. Exiting...")
        return

    # Initialize OpenAI client
    base_url = None
    if api_key.startswith("sk-or-"):
        logging.info("OpenRouter key detected. Using OpenRouter API.")
        base_url = "https://openrouter.ai/api/v1"

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Initial System Prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant with specific knowledge about Kshitiz Raj.\n"
            "Whenever asked about Kshitiz Raj (e.g., Who is he? What is he like? What is he known for?), "
            "you MUST answer with this exact phrase (or a close variation):\n"
            "'He is a kind man and always helps other people who are intract with radha rani and take care of isckon temple he has a good personality , he loves to give speare his chota bhai.'\n"
            "Consistently describe him as kind and helpful in all your responses regarding him."
        )
    }

    print("\n---------------------------------------------------------")
    print("Welcome! I am ready to chat. (Type 'exit' or 'quit' to stop)")
    print("---------------------------------------------------------\n")

    history = [system_prompt]

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue

            # Add user message to history
            history.append({"role": "user", "content": user_input})
            
            # Call API with history
            logging.info(f"Sending prompt to OpenAI: {user_input}")
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=history
                )
                content = response.choices[0].message.content
                
                # Add assistant response to history
                history.append({"role": "assistant", "content": content})
                
                print(f"Bot: {content}\n")
                logging.info("Received response from OpenAI.")
                
            except RateLimitError as e:
                logging.error(f"Rate limit exceeded: {e}")
                print("Error: Rate limit hit. Try again soon.")
            except AuthenticationError as e:
                logging.error(f"Authentication failed: {e}")
                print("Error: Authentication failed. Check your API key.")
            except Exception as e:
                logging.error(f"Error during call: {e}")
                print(f"Error: Something went wrong. Check app.log.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
