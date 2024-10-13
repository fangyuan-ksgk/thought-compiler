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
    img: np.ndarray
    pdf_path: str

    def save(self, output_dir: str):
        
        # Convert numpy array to base64 string
        if isinstance(self.img, np.ndarray):
            img_array = self.img
            img = Image.fromarray(self.img)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            self.img = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        with open(os.path.join(output_dir, f"{self.title}.json"), "w") as f:
            json.dump(asdict(self), f)
            
        self.img = img_array
        
    @classmethod
    def load(cls, path: str) -> 'Paper':
        with open(path, "r") as f:
            data = json.load(f)
        
        # Convert base64 string back to numpy array
        if 'img' in data and isinstance(data['img'], str):
            img_data = base64.b64decode(data['img'])
            img = Image.open(io.BytesIO(img_data))
            data['img'] = np.array(img)
        
        return cls(**data)
    
    
def get_paper_info(r: arxiv.arxiv.Result):
    paper_info = {
        "title": r.title,
        "summary": r.summary,
        "tags": [cat.lower() for cat in r.categories],
        "citations": [], # TODO: get citations by parsing PDF 
        "date": r.published.strftime("%Y-%m-%d"),
        "img": None,
        "pdf_path": None
    }
    os.makedirs("cave/paper", exist_ok=True)
    paper_path = r.download_pdf(dirpath="cave/paper/")
    citations, tags, img = extract_citations_and_tags(paper_path)
    os.remove(paper_path)
    paper_info["citations"] = citations
    paper_info["tags"] = tags
    paper_info["img"] = img
    paper_info["pdf_path"] = paper_path
    return paper_info

def get_paper_info_batch(search_results: List[arxiv.arxiv.Result], vllm: VLLM):
    os.makedirs("cave/paper", exist_ok=True)
    paper_paths = []
    paper_infos = []

    # Download PDFs and prepare basic info
    for r in tqdm(search_results, desc="Downloading PDFs"):
        paper_path = r.download_pdf(dirpath="cave/paper/")
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
    
    tag_responses = vllm.generate(tag_prompts)
    cite_responses = vllm.generate(cite_prompts)
    tags = [parse_tags(response) for response in tag_responses]
    citations = [parse_citations(response) for response in cite_responses]

    return tags, citations, imgs


    
