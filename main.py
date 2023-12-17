from gpt import ChatGPT
import asyncio

async def wait_test(gpt, message):
    try:
        #Fake Async Server Call
        await asyncio.wait_for(gpt.async_request_function(message), timeout=5)
        return True
    except asyncio.TimeoutError:
        print("TimeoutError")
        # sys.exit(1)
        return False
    
def main(gpt):
    while True:
        input_text = str(input())
        if input_text == "exit":
            break
        while True:
            done = asyncio.run(wait_test(gpt, input_text))
            if done:
                break
            else:
                print("Re-try the request")
                continue
        response = gpt.get_response()
        print(response)
        prompt_tokens, completion_tokens, price = gpt.count_price()
        print(f"Token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_price={price}")
    gpt.reset()
    # If you call reset, you can start a new conversation with GPT, but the previous conversation will be lost.
    # You also need to re-configure the GPT system and schema.

if __name__ == "__main__":
    # Configure the GPT chatbot
    organization = "<Your organization>"
    api_key = "<Your api key.>"
    gpt = ChatGPT(organization=organization, api_key=api_key, model='gpt-3.5-turbo')
    
    # Configure your GPT characteristics
    system = '''
    You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\n
    Knowledge cutoff: 2023-04\n
    Current date: 2023-11-03\n\n 
    '''
    
    schema = {
        "type": "object",
        "properties": {
            "key_visual_elements": {
                "type": "array",
                "description": "A list of key visual elements in the query.",
                "items": {"type": "string"},
            },
            "explaination": {
                "type": "array",
                "description": "A breif reasoning of why the predicted span is relevant to the query.",
                "items": {"type": "string"},
            },
            "relevant_span": {
                "type": "array",
                "description": "A start and end frame of span that are consistent with the query.",
                "items": {"type": "number"},
            },
        },
        "required": ["key_visual_elements", "explaination", "relevant_span"],
        # "required": ["relevant_span"],
    }
    gpt.set_system(system=system)
    gpt.set_schema(schema=schema)
    
    # Start to chat with GPT
    print("Start to chat with GPT. Type 'exit' to exit.")
    print("Type something to chat with GPT: ")

    main(gpt)