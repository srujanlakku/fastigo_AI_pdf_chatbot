import inspect
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

print('ChatGoogleGenerativeAI init:', inspect.signature(ChatGoogleGenerativeAI.__init__))
print('ChatGoogleGenerativeAI generate:', inspect.signature(ChatGoogleGenerativeAI.generate))
print('ChatGoogleGenerativeAI stream:', inspect.signature(ChatGoogleGenerativeAI.stream))
print('ChatGoogleGenerativeAI stream_events:', inspect.signature(ChatGoogleGenerativeAI.stream_events))
print('ChatGoogleGenerativeAI methods:', [m for m in dir(ChatGoogleGenerativeAI) if 'stream' in m or 'generate' in m])
