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

oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_last_dates(days=5):
    return [(datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

# save the quick info into a json file 
def save_quick_info(quick_info, filename="cave/arxiv_papers_info.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(quick_info, f, ensure_ascii=False, indent=4)
        
def read_quick_info(filename="cave/arxiv_papers_info.json"):
    with open(filename, 'r', encoding='utf-8') as f:
        quick_info = json.load(f)
    return quick_info
        
def crawl_arxiv_papers(query="AI", max_results=200):
    # Construct the default API client.
    client = arxiv.Client()

    # Search for the 10 most recent articles matching the keyword "quantum."
    search = arxiv.Search(
    query = query,
    max_results = max_results,
    sort_by = arxiv.SortCriterion.SubmittedDate
    )

    results = client.results(search)

    # `results` is a generator; you can iterate over its elements one by one...
    quick_info = []
    for r in tqdm(client.results(search), desc = "Downloading Newest Arxiv Papers"):
        quick_info.append({"title": r.title, "summary": r.summary})
        # Decision spot: whether to download the paper locally or skip the summary 
        r.download_pdf(dirpath="cave/paper")

    # Call this function after populating quick_info
    save_quick_info(quick_info)
    print("ArXiv paper crawling completed!")

    
    
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
    # Convert the first page of the PDF to an image
    images = convert_from_path(pdf_file, first_page=first_page, last_page=last_page)

    pdf_base64_images = []
    for pdf_image in images:
        # Convert PIL Image to base64 string
        buffered = io.BytesIO()
        pdf_image.save(buffered, format="PNG")
        pdf_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        pdf_base64_images.append(pdf_image_base64)
        
    return pdf_base64_images


from typing import TypedDict, List

class Paper(TypedDict):
    title: str
    summary: str
    post: str
    evaluation: str
    
def add_papers(left: List[Paper], right: List[Paper]) -> List[Paper]:
    return left + right 
    
def initialize_papers(papers: list) -> list[Paper]:
    paper_list = []
    for paper in papers:
        
        paper_list.append(Paper(title=paper.get("title"), 
                                summary=paper.get("summary"), 
                                post=paper.get("post", ""), 
                                comment=paper.get("comment", ""),
                                score=paper.get("score", 0)))
    return paper_list
    
def load_papers(refresh: bool = False, file_path: str = "cave/arxiv_papers_info.json") -> List[Paper]:
    if refresh or not os.path.exists(file_path):
        crawl_arxiv_papers()
    papers = read_quick_info()
    return initialize_papers(papers)

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
    return return_str.strip()
    
def papers_to_string(papers: List[Paper]) -> str:
    return "\n\n".join([paper_to_string(paper) for paper in papers])


def parse_paper_response(content: str) -> List[Paper]:
    # Extract content inside square brackets
    match = re.search(r'\[(.*?)\]', content, re.DOTALL)
    if not match:
        print("Error: No JSON array found in square brackets")
        return []
    
    json_content = match.group(1)
    
    try:
        # Parse the JSON content
        paper_data = json.loads(f"[{json_content}]")
        
        # Create a list to store Paper objects
        parsed_papers = []
        
        for item in paper_data:
            # Create a Paper object for each item in the JSON
            paper = Paper(
                title=item['title'],
                comment=item['comment'],
                score=item['score']
            )
            parsed_papers.append(paper)
        
        return parsed_papers
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the response")
        return []
    except KeyError as e:
        print(f"Error: Missing key in JSON data: {e}")
        return []

# Json helper functions


