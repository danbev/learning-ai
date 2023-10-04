from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# llama.cpp has been updated to handle GGUF format instead of GGML format which
# means that we need to convert the older GGML format to GGUF format:
# $ ~/work/ai/llama.cpp/convert-llama-ggml-to-gguf.py --input models/llama-2-7b-chat.ggmlv3.q2_K.bin  --output models/llama-2-7b-chat.gguf.q2_K.bin

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.gguf.q2_K.bin",
    temperature=0,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True, # Verbose is required to pass to the callback manager
)
#print(llm)

prompt = """
Question: Can you give me a short description of how Austin Danger Powers is?"
"""
llm(prompt)


