from openai import AzureOpenAI
import os

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.environ.get("AZURE_OPENAI_API_BASE")
)

MODEL_NAME = "gpt-5"

# Define a function to interact with gpt
def ask_gpt(prompt):
    try:
        # Call the OpenAI API to generate a response
      completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
          ]
        )

      return completion.choices[0].message.content

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example prompt
user_prompt = "What is the capital of France?"

# Get GPT's response
gpt_response = ask_gpt(user_prompt)

# Print the response
print("GPT:", gpt_response)