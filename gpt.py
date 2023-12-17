import os, sys
import openai
from openai import OpenAI, AsyncOpenAI
import asyncio


# Example input:
# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Who won the world series in 2020?"},
#         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#         {"role": "user", "content": "Where was it played?"}
#     ]
# )

# Example output:
# {
#   "id": "chatcmpl-7xR2zcCPnrhSfV9khD1ab6SQmKGJ3",
#   "object": "chat.completion",
#   "created": 1694399677,
#   "model": "gpt-3.5-turbo-0613",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "The 2020 World Series was played at Globe Life Field in Arlington, Texas."
#       },
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 53,
#     "completion_tokens": 17,
#     "total_tokens": 70
#   }
# }


class ChatGPT:
    def __init__(self, organization, api_key, do_async=False, schema={}, model="gpt-3.5-turbo", seed=1126, max_tokens=1000, temperature=0.1):
        self.organization = organization
        self.api_key = api_key
        self.model = model
        self.do_async = do_async
        # self.client.organization = organization
        # self.client.api_key = api_key
        if self.do_async:
            self.client = AsyncOpenAI(api_key=api_key, organization=organization)
        else:
            self.client = OpenAI(api_key=api_key, organization=organization)
        self._messages = []
        self._stored_responses = []
        self._stored_requests = []
        self.pricing_dict = {
            'gpt-3.5-turbo': (0.0015, 0.002),
            'gpt-3.5-turbo-1106': (0.001, 0.002),
            'gpt-3.5-turbo-instruct': (0.0015, 0.002),
            'gpt-3.5-turbo-16k': (0.003, 0.004),
            'gpt-4': (0.03, 0.06),
            'gpt-4-32k': (0.06, 0.12),
            'gpt-4-0314': (0.03, 0.06),
            'gpt-4-0613': (0.03, 0.06),
            'gpt-4-1106-preview': (0.01, 0.03),
            'gpt-4-1106-vision-preview': (0.01, 0.03),
        }
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.schema = schema
        self._response = None

    def set_system(self, system = "You are a helpful assistant."):
        self.system = system
        self._messages.append({"role": "system", "content": system})
    
    def set_schema(self, schema):
        self.schema = schema

    def request(self, message):
        self._messages.append({"role": "user", "content": message})
        self._stored_requests.append(message)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            seed = self.seed,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
        )
        print(self._messages)
        print(response)
        self._response = response.choices[0].message.content
        self._stored_responses.append(response)
        self._messages.append(response.choices[0].message)
        return response.choices[0].message.content
    
    async def async_request(self, message):
        self._messages.append({"role": "user", "content": message})
        self._stored_requests.append(message)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            seed = self.seed,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
        )
        print(self._messages)
        print(response)
        self._response = response.choices[0].message.content
        self._stored_responses.append(response)
        self._messages.append(response.choices[0].message)
        return response.choices[0].message.content

    def request_function(self, message):
        self._messages.append({"role": "user", "content": message})
        self._stored_requests.append(message)
        response = self.client.with_options(max_retries=5).chat.completions.create(
            model=self.model,
            messages=self._messages,
            seed = self.seed,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
            functions = [{"name": "Retrieval", "parameters": self.schema}],
            function_call = {"name": "Retrieval"},
        )
        print(self._messages)
        print(response)
        self._response = response.choices[0].message.function_call.arguments
        self._stored_responses.append(response)
        self._messages.append(response.choices[0].message)
        return response.choices[0].message.function_call.arguments

    async def async_request_function(self, message):
        self._messages.append({"role": "user", "content": message})
        self._stored_requests.append(message)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            seed = self.seed,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
            functions = [{"name": "Retrieval", "parameters": self.schema}],
            function_call = {"name": "Retrieval"},
        )
        print(self._messages)
        print(response)
        self._response = response.choices[0].message.function_call.arguments
        self._stored_responses.append(response)
        self._messages.append(response.choices[0].message)
        return response.choices[0].message.function_call.arguments
    
    def reset(self):
        self._messages = []
        self._stored_responses = []
        self._stored_requests = []
        self.schema = {}
        self._response = None
    
    def get_all_requests(self):
        return self._stored_requests
    
    def get_all_responses(self):
        return self._stored_responses
    
    def get_all_messages(self):
        return self._messages
    
    def get_response(self):
        return self._response
    
    def count_usage(self):
        # collect the total number of tokens used in _stored_responses
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        for response in self._stored_responses:
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            total_tokens += response.usage.total_tokens
        return prompt_tokens, completion_tokens, total_tokens
    
    def count_price(self):
        price = 0
        prompt_tokens, completion_tokens, total_tokens = self.count_usage()
        input_price, output_price = self.price()
        price += input_price * prompt_tokens / 1000
        price += output_price * completion_tokens / 1000
        return prompt_tokens, completion_tokens, price
    
    def price(self):
        return self.pricing_dict[self.model]

# async def wait_test(gpt, message):
#     try:
#         #Fake Async Server Call
#         await asyncio.wait_for(gpt.async_request_function(message), timeout=5)
#         return True
#     except asyncio.TimeoutError:
#         print("TimeoutError")
#         # sys.exit(1)
#         return False

# if __name__ == "__main__":
#     # Configure the GPT chatbot
#     organization = "<Your organization>"
#     api_key = "<Your api key.>"
#     gpt = ChatGPT(organization=organization, api_key=api_key, model='gpt-3.5-turbo')
    
#     # Configure your GPT characteristics
#     system = '''
#     You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\n
#     Knowledge cutoff: 2023-04\n
#     Current date: 2023-11-03\n\n 
#     '''
    
#     schema = {
#         "type": "object",
#         "properties": {
#             "key_visual_elements": {
#                 "type": "array",
#                 "description": "A list of key visual elements in the query.",
#                 "items": {"type": "string"},
#             },
#             "explaination": {
#                 "type": "array",
#                 "description": "A breif reasoning of why the predicted span is relevant to the query.",
#                 "items": {"type": "string"},
#             },
#             "relevant_span": {
#                 "type": "array",
#                 "description": "A start and end frame of span that are consistent with the query.",
#                 "items": {"type": "number"},
#             },
#         },
#         "required": ["key_visual_elements", "explaination", "relevant_span"],
#         # "required": ["relevant_span"],
#     }
#     gpt.set_system(system=system)
#     gpt.set_schema(schema=schema)
    
#     # Start to chat with GPT
#     print("Start to chat with GPT. Type 'exit' to exit.")
#     print("Type something to chat with GPT: ")

#     while True:
#         input_text = str(input())
#         if input_text == "exit":
#             break
#         while True:
#             done = asyncio.run(wait_test(gpt, input_text))
#             if done:
#                 break
#             else:
#                 print("Re-try the request")
#                 continue
#         response = gpt.get_response()
#         print(response)
#         # respone = gpt.request(input_text)
#         # respone = gpt.request_function(input_text)
#         # prompt_tokens, completion_tokens, total_tokens = gpt.count_usage()
#         prompt_tokens, completion_tokens, price = gpt.count_price()
#         print(f"Token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_price={price}")
#     gpt.reset()

    