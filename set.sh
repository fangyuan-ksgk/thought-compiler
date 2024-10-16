brew install poppler anthropic openai arxiv fitz frontend
pip install pdf2image
pip install -U langgraph
pip install --upgrade transformers trl huggingface_hub datasets accelerate bitsandbytes peft vllm deepspeed
MAX_JOBS=4 pip install flash-attn -U --no-build-isolation --force-reinstall
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3