from typing import TypedDict, List
import arxiv 
from tqdm import tqdm
import re
import datetime
import json

def get_last_dates(days=5):
    return [(datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

def save_quick_info(quick_info, filename="cave/arxiv_papers_info.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(quick_info, f, ensure_ascii=False, indent=4)
        
        
class Paper(TypedDict):
    title: str
    summary: str
    post: str
    comment: str
    score: int
    argument: str
    tags: List[str]
    citations: List[str]
    
def crawl_arxiv_papers(query="AI", max_results=200):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    quick_info = []
    for r in tqdm(client.results(search), desc="Downloading Newest Arxiv Papers"):
        paper_info = {
            "title": r.title,
            "summary": r.summary,
            "tags": [cat.lower() for cat in r.categories],
            "citations": extract_citations(r.summary)
        }
        quick_info.append(paper_info)
        r.download_pdf(dirpath="cave/paper")

    save_quick_info(quick_info)
    print("ArXiv paper crawling completed!")

def extract_citations(text):
    citation_pattern = r'\[([^\]]+)\]'
    return re.findall(citation_pattern, text)