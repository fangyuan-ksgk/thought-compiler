import anthropic
from typing import Union, List
from openai import OpenAI
import os

anthropic_client = anthropic.Anthropic()
openai_client = OpenAI()

def get_text_content(query: Union[str, List[str]]):
    if isinstance(query, list):
        return [{"type": "text", "text": q} for q in query]
    else:
        return [{"type": "text", "text": query}]

def get_claude_response(query: Union[str, List[str]], img: str = None, img_type: str = None, system_prompt: str = "You are a helpful assistant."):
    """ 
    Claude response with query and image input
    """
    text_content = get_text_content(query)
    
    if img is not None:
        img_content = [{"type": "image", "source": {"type": "base64", "media_type": img_type, "data": img}}]
        content = img_content + text_content
    else:
        content = text_content
    
    message = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        system=system_prompt,
    )
    return message.content[0].text


def get_openai_response(query: Union[str, List[str]], img: str = None, img_type: str = "image/jpeg", system_prompt: str = "You are a helpful assistant."):
    """
    OpenAI response with query and optional image input
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    content = []
    if isinstance(query, str):
        content.append({"type": "text", "text": query})
    elif isinstance(query, list):
        for q in query:
            if isinstance(q, dict) and "role" in q and "content" in q:
                content.append({"type": "text", "text": q["content"]})
            else:
                content.append({"type": "text", "text": q})
    
    if img is not None:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img_type};base64,{img}"
            }
        })
    
    messages.append({"role": "user", "content": content})
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1024
    )
    return response.choices[0].message.content


from groq import Groq

groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def get_groq_response(prompt: str):
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.2-3b-preview",
    )

    return chat_completion.choices[0].message.content

import torch
from typing import Optional
import os 
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    class VLLM:
        def __init__(
            self,
            name: str,
            # gpu_ids: List[int] = [0, 1], # Assuming we have 2 GPUs here
            download_dir: Optional[str] = None,
            dtype: str = "auto",
            gpu_memory_utilization: float = 0.85,
            max_model_len: int = 4096,
            merge: bool = False,
            **kwargs,
        ) -> None:
            self.name: str = name
            if merge:
                os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN" # Use this for merged model
            else:
                # os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" # Use this for baseline model | Gemma2 require this backend for inference
                os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN" # Default to using llama3.1 70B (quantized version, of course)            
            
            available_gpus = list(range(torch.cuda.device_count()))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
            
            # if len(available_gpus) > 1:
            #     import multiprocessing
            #     multiprocessing.set_start_method('spawn', force=True)

            self.model: LLM = LLM(
                model=self.name,
                tensor_parallel_size=len(available_gpus),
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                download_dir=download_dir,
                max_model_len=max_model_len,
            )
            
            self.params = SamplingParams(**kwargs)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        
        def completions(
            self,
            prompts: List[str],
            use_tqdm: bool = False,
            **kwargs: Union[int, float, str],
        ) -> List[str]:
            formatted_prompts = [self.format_query_prompt(prompt.strip()) for prompt in prompts]
    
            outputs = self.model.generate(formatted_prompts, self.params, use_tqdm=use_tqdm)
            outputs = [output.outputs[0].text for output in outputs]
            return outputs

        def generate(
            self,
            prompts: List[str],
            use_tqdm: bool = False,
            max_new_tokens: int = 2048,
            **kwargs: Union[int, float, str],
        ) -> List[str]:
            formatted_prompts = [self.format_query_prompt(prompt.strip()) for prompt in prompts]
            return self.model.generate(formatted_prompts, self.params, use_tqdm=use_tqdm)

        def format_query_prompt(self, prompt: str, completion: str = "####Dummy-Answer") -> str:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            format_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            query_prompt = format_prompt.split(completion)[0]
            return query_prompt
        
except ImportError:
    class VLLM:
        def __init__(self, *args, **kwargs):
            pass
        def completions(self, *args, **kwargs):
            return get_openai_response(*args, **kwargs)
    
    # Just write a dummy VLLM class for Mac instance here 
    print("Could not load vllm class, check CUDA support and GPU RAM size")