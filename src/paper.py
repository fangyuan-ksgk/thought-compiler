from typing import TypedDict, List
import arxiv 
from tqdm import tqdm
import re
import datetime
import json
import numpy as np
from dataclasses import dataclass, asdict
from .utils import *
from PIL import Image
import io
import base64
import os
from .llm import *
from .prompt import *

 
MAX_ATTEMPTS = 3

@dataclass
class Paper:
    title: str
    summary: str
    tags: List[str]
    citations: List[str]
    date: str
    pdf_path: str

    def save(self, output_dir: str, name: Optional[str] = None):
        if name is None:
            name = self.title
        with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
            json.dump(asdict(self), f)
        
    @classmethod
    def load(cls, path: str) -> 'Paper':
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls(**data)

from typing import Callable

def get_paper_info_without_tags(r):
    paper_info = {
        "title": r.title,
        "summary": r.summary,
        "tags": [],
        "citations": [], # TODO: get citations by parsing PDF 
        "date": r.published.strftime("%Y-%m-%d"),
        "pdf_path": ""
    }
    os.makedirs("cave/paper", exist_ok=True)
    # tags = extract_tags(r.summary, get_llm_response)
    # paper_info["tags"] = tags
    return paper_info
    
def get_paper_info(r, get_llm_response: Callable):
    paper_info = {
        "title": r.title,
        "summary": r.summary,
        "tags": [],
        "citations": [], # TODO: get citations by parsing PDF 
        "date": r.published.strftime("%Y-%m-%d"),
        "pdf_path": ""
    }
    os.makedirs("cave/paper", exist_ok=True)
    tags = extract_tags(r.summary, get_llm_response)
    paper_info["tags"] = tags
    return paper_info


def get_paper_info_batch(search_results: list, vllm: VLLM):
    os.makedirs("cave/paper", exist_ok=True)
    paper_paths = []
    paper_infos = []

    # Download PDFs and prepare basic info
    for r in tqdm(search_results, desc="Downloading PDFs"):
        try:
            paper_path = r.download_pdf(dirpath="cave/paper/")
        except:
            continue
        paper_paths.append(paper_path)
        paper_infos.append({
            "title": r.title,
            "summary": r.summary,
            "tags": [],
            "citations": [],
            "date": r.published.strftime("%Y-%m-%d"),
            "img": None,
            "pdf_path": paper_path
        })

    # Batch extract citations and tags
    tags, citations, imgs = extract_citations_and_tags_batch(paper_paths, vllm)

    # Update paper_infos with extracted data
    for i, paper_info in enumerate(paper_infos):
        paper_info["tags"].extend(tags[i])
        paper_info["citations"] = citations[i]
        paper_info["img"] = imgs[i]

    # Clean up downloaded PDFs
    for paper_path in paper_paths:
        os.remove(paper_path)

    return paper_infos

from typing import Callable

def extract_tags(summary: str, get_llm_response: Callable):
    try:
        response = get_llm_response(EXTRACT_TAGS_PROMPT + summary, system_prompt="You are an AI assistant specialized in analyzing academic papers. Provide output in JSON format only.")
        tags = parse_tags(response)
    except Exception as e:
        tags = []
    return tags
    
def extract_citations_and_tags(paper_path: str):
    """ 
    LLM-based citations and tags extraction
    """

    texts, img = pdf_to_text_and_images(paper_path) # Page-Image and Text of each page of the paper
    full_text = " ".join(texts)

    has_result = False
    attempts = 0
    while not has_result and attempts < MAX_ATTEMPTS:
        try:
            response = get_openai_response(EXTRACT_CITATIONS_AND_TAGS_PROMPT + full_text, system_prompt="You are an AI assistant specialized in analyzing academic papers. Provide output in JSON format only.")
            citations, tags = parse_citation_and_tags(response)
            has_result = len(citations) > 0 and len(tags) > 0
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Issue response: \n", response)
            citations, tags = [], []
            has_result = False
        attempts += 1
    
    return citations, tags, img


def extract_citations_and_tags_batch(paper_paths: List[str], vllm: VLLM):
    """ 
    LLM-based citations and tags extraction
    - use vLLM
    - use 1st page txt for tags extraction
    - use 5 pages starting from the first occurance of "References" for citations extraction
    """

    # Extract relevant Raw Materials
    tag_txts, cite_txts, imgs = [], [], []
    for paper_path in paper_paths:
        texts, img = pdf_to_text_and_images(paper_path) # Page-Image and Text of each page of the paper
        full_text = " ".join(texts)
        tag_txt = texts[0]
        cite_txt = "".join(texts[texts.index(next(t for t in texts if "References" in t)):texts.index(next(t for t in texts if "References" in t))+5])
        tag_txts.append(tag_txt)
        cite_txts.append(cite_txt)
        imgs.append(img)
        
    # Batch Inference with vLLM
    tag_prompts = [EXTRACT_TAGS_PROMPT + tag_txt for tag_txt in tag_txts]
    cite_prompts = [EXTRACT_CITATIONS_PROMPT + cite_txt for cite_txt in cite_txts]
    
    tag_responses = vllm.completions(tag_prompts, use_tqdm=True)
    cite_responses = vllm.completions(cite_prompts, use_tqdm=True)
    tags = [parse_tags(response) for response in tag_responses]
    citations = [parse_citations(response) for response in cite_responses]

    return tags, citations, imgs

    
