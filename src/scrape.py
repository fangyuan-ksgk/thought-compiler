from pdf2image import convert_from_path
from tqdm import tqdm as tqdm 
from openai import OpenAI 
import datetime
import base64
import arxiv
import json
import os
import io
import re
from langgraph.graph.message import add_messages
import random
from typing import Annotated, Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from .paper import Paper

oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = ChatOpenAI(model="gpt-4o")

def get_last_dates(days=5):
    return [(datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

def save_quick_info(quick_info, filename="cave/arxiv_papers_info.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(quick_info, f, ensure_ascii=False, indent=4)
        
def read_quick_info(filename="cave/arxiv_papers_info.json"):
    with open(filename, 'r', encoding='utf-8') as f:
        quick_info = json.load(f)
    return quick_info

def get_oai_response(prompt, system_prompt="You are a helpful assistant", img=None, img_type=None):
    if isinstance(prompt, str):
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        msg = [
            {"role": "system", "content": system_prompt},
        ]
        msg.extend(prompt)
    
    if img is not None and img_type is not None:
        if isinstance(img, str):
            img = [img]
        image_content = []
        for _img in img:
            image_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_img}"}})
            
        text = msg[-1]["content"]
        text_content = [{"type": "text", "text": text}]
        
        msg.append({
            "role": "user",
            "content": text_content + image_content,
        })
        
    response = oai_client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=msg,
    )
    
    print("Response: ", response.choices[0].message.content)
    
    return response.choices[0].message.content

def get_pdf_contents(pdf_file, first_page=1, last_page=1):
    images = convert_from_path(pdf_file, first_page=first_page, last_page=last_page)

    pdf_base64_images = []
    for pdf_image in images:
        buffered = io.BytesIO()
        pdf_image.save(buffered, format="PNG")
        pdf_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        pdf_base64_images.append(pdf_image_base64)
        
    return pdf_base64_images
    
def add_papers(left: List[Paper], right: List[Paper]) -> List[Paper]:
    merged = {paper['title']: paper for paper in left}
    for paper in right:
        if paper['title'] in merged:
            merged[paper['title']] = Paper(
                title=paper['title'],
                summary=paper['summary'] if paper['summary'] else merged[paper['title']]['summary'],
                post=paper['post'] if paper['post'] else merged[paper['title']]['post'],
                comment=merged[paper['title']]['comment'] if merged[paper['title']]['comment'] else paper['comment'],
                score=(merged[paper['title']]['score'] + paper['score']) / 2 if merged[paper['title']]['score'] != 0 and paper['score'] != 0 else max(merged[paper['title']]['score'], paper['score']),
                argument=merged[paper['title']]['argument'] if merged[paper['title']]['argument'] else paper['argument'],
                tags=list(set(merged[paper['title']]['tags'] + paper['tags'])),
                citations=list(set(merged[paper['title']]['citations'] + paper['citations']))
            )
        else:
            merged[paper['title']] = paper
    return list(merged.values())

class State(TypedDict):
    messages: Annotated[list, add_messages]
    papers: Annotated[list, add_papers]
    
def initialize_papers(papers: list) -> list[Paper]:
    paper_list = []
    for paper in papers:
        paper_list.append(Paper(
            title=paper.get("title"),
            summary=paper.get("summary"),
            post=paper.get("post", ""),
            comment=paper.get("comment", ""),
            score=paper.get("score", 0),
            argument=paper.get("argument", ""),
            tags=paper.get("tags", []),
            citations=paper.get("citations", [])
        ))
    return paper_list
    


def paper_to_string(paper: Paper) -> str:
    return_str = ""
    if paper.get('title'):
        return_str += f"Title: {paper['title']}\n"
    if paper.get('summary'):
        return_str += f"Summary: {paper['summary']}\n"
    if paper.get('post'):
        return_str += f"Post: {paper['post']}\n"
    if paper.get('comment'):
        return_str += f"Comment: {paper['comment']}\n"
    if paper.get('score') is not None:
        return_str += f"Score: {paper['score']}\n"
    if paper.get('argument'):
        return_str += f"Argument: {paper['argument']}\n"
    if paper.get('tags'):
        return_str += f"Tags: {', '.join(paper['tags'])}\n"
    if paper.get('citations'):
        return_str += f"Citations: {', '.join(paper['citations'])}\n"
    return return_str.strip()
    
def papers_to_string(papers: List[Paper]) -> str:
    return "\n\n".join([paper_to_string(paper) for paper in papers])

import ast 

def load_json_with_ast(json_str):
    json_str_cleaned = json_str.strip()
    papers = ast.literal_eval(json_str_cleaned)
    return papers

def parse_json_response(content):
    match = re.search(r'\[(.*?)\]', content, re.DOTALL)
    json_content = match.group(1)

    json_str = f"[{json_content}]"

    try: 
        json_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        try:
            json_data = load_json_with_ast(json_str)
        except:
            return []
    return json_data

def parse_paper_response(content: str) -> List[Paper]:
    content = content.strip()
    
    paper_data = parse_json_response(content)
    
    if not paper_data:
        return []
    
    parsed_papers = []
    
    for item in paper_data:
        try:
            paper = Paper(
                title=item['title'],
                summary=item.get('summary', ''),
                post=item.get('post', ''),
                comment=item['comment'],
                score=item['score'],
                argument=item.get('argument', ''),
                tags=item.get('tags', []),
                citations=item.get('citations', [])
            )
            parsed_papers.append(paper)
        except KeyError as e:
            print(f"Error: Missing key in JSON data: {e}")
    
    return parsed_papers