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
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = ChatOpenAI(model="gpt-4o")

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
import random
from typing import Annotated, Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class Paper(TypedDict):
    title: str
    summary: str
    post: str
    comment: str
    score: int
    argument: str # Human feedback on the paper 
    
def add_papers(left: List[Paper], right: List[Paper]) -> List[Paper]:
    # When there are duplicate titles, we merge, otherwise we concatenate
    merged = {paper['title']: paper for paper in left}
    for paper in right:
        if paper['title'] in merged:
            # Merge the papers with the same title
            merged[paper['title']] = Paper(
                title=paper['title'],
                summary=paper['summary'] if paper['summary'] else merged[paper['title']]['summary'],
                post=paper['post'] if paper['post'] else merged[paper['title']]['post'],
                comment=merged[paper['title']]['comment'] if merged[paper['title']]['comment'] else paper['comment'],
                score=(merged[paper['title']]['score'] + paper['score']) / 2 if merged[paper['title']]['score'] != 0 and paper['score'] != 0 else max(merged[paper['title']]['score'], paper['score']),
                argument=merged[paper['title']]['argument'] if merged[paper['title']]['argument'] else paper['argument']
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
        
        paper_list.append(Paper(title=paper.get("title"), 
                                summary=paper.get("summary"), 
                                post=paper.get("post", ""), 
                                comment=paper.get("comment", ""),
                                score=paper.get("score", 0),
                                argument=paper.get("argument", "")
                                ))
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
    if paper.get('argument') is not None:
        return_str += f"Argument: {paper['argument']}\n"
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
    # Remove leading/trailing whitespace
    content = content.strip()
    
    paper_data = parse_json_response(content)
    
    if not paper_data:
        return []
    
    # Create a list to store Paper objects
    parsed_papers = []
    
    for item in paper_data:
        try:
            # Create a Paper object for each item in the JSON
            paper = Paper(
                title=item['title'],
                summary=item.get('summary', ''),  # Use get() with default value
                post=item.get('post', ''),
                comment=item['comment'],
                score=item['score'],
                argument=item.get('argument', '')
            )
            parsed_papers.append(paper)
        except KeyError as e:
            print(f"Error: Missing key in JSON data: {e}")
            # Continue processing other items even if one fails
    
    return parsed_papers

# Json helper functions

SELECT_PAPER_PROMPT = """Here is a list of papers with their summaries. \n{paper_string}\nFor each paper, conduct an analysis, provide your comment on the contribution and limitations of the paper, as well as a score between 0 and 100 indicating its importance.
Your response should be in a JSON format as a list of dictionaries. Each dictionary should have the following keys: title, comment, score. For example:
[
    {{
        "title": "Paper Title 1",
        "comment": "Comment on the importance and contribution of the paper",
        "score": 80
    }},
    {{
        "title": "Paper Title 2",
        "comment": "Another comment on a different paper",
        "score": 75
    }}
]

Please provide your analysis for each paper in the list."""


ARGUMENT_PROMPT = """Provide argument on why the paper {paper_title} is the best among the selected papers: {paper_string}"""


class CrawlNode:
    def __init__(self, name: str, refresh: bool = False):
        self.name = name
        self.path = "cave/arxiv_papers_info.json"
        self.refresh = refresh
    def __call__(self, state: State): # Wonderful, now it works (!) I am starting to like LangGraph now ...
        list_of_papers = load_papers(refresh=self.refresh, file_path=self.path)
        return {"papers": list_of_papers}
    

class SelectNode:
    def __init__(self, name: str,batch_id: int, model: ChatOpenAI):
        self.name = name
        self.model = model
        self.batch_id = batch_id

    def __call__(self, state: State):
        papers = state["papers"]
        batch_size = 10

        start_index = self.batch_id * batch_size
        end_index = min((self.batch_id + 1) * batch_size, len(papers))

        batch_papers = papers[start_index:end_index]
        paper_string = papers_to_string(batch_papers)
        select_prompt = SELECT_PAPER_PROMPT.format(paper_string=paper_string)

        paper_response = []
        max_tries = 3
        cur_try = 0
        while not paper_response and cur_try < max_tries:
            response = self.model.invoke(select_prompt)
            paper_response = parse_paper_response(response.content)
            cur_try += 1
        
        if not paper_response:
            return {"papers": batch_papers}
        
        processed_batch = initialize_papers(paper_response)
        batch_papers = add_papers(batch_papers, processed_batch) 

        # Update the state with the processed and verified papers
        return {"papers": batch_papers}
    
# Understanding Human Preference 

REFLECT_PROMPT = """Base on human feedback on AI papers, provide your understanding of his preference and how can you better select papers which he likes. 
AI comment on paper: {summary}
Human comment on paper: {feedback}
Provide concise description within 200 words."""

SUMMARIZE_PROMPT = """Please summarize the following information into a concise description within 200 words. 
{reflection_str}"""

# After Human Feedback is collected, we analyze and summarize their preference for future paper selection
# Such understanding of his preference will be passed around over all the AI workers

def summarize_human_preference(papers, model):
    # now we are talking about customization (;->)
    reflections = []
    for paper in papers:
        if not paper["argument"]:
            continue 
        feedback, summary = paper["argument"], paper["comment"] 
        reflect_prompt = REFLECT_PROMPT.format(summary=summary, feedback=feedback)
        reflection = model.invoke(reflect_prompt)
        reflections.append(reflection)

    reflection_str = ("\n").join([r.content for r in reflections])

    summary_prompt = SUMMARIZE_PROMPT.format(reflection_str=reflection_str)
    summary = model.invoke(summary_prompt)
    return summary.content

from typing import Callable
    
    

def add_selection_nodes(builder, current_node, next_node, model):

    # Number of nodes to create at this level
    node_names = [f"Paper_Batch_{i}" for i in range(5)]
    for i, nm in enumerate(node_names):
        node_name = f"Crawl_{nm}"
        builder.add_node(node_name, SelectNode(node_name, i, model))
        builder.add_edge(current_node, node_name)

    # Connecting all branches back to the END node
    for node_name in node_names:
        crawl_node_name = f"Crawl_{node_name}"
        builder.add_edge(crawl_node_name, next_node)
        
    return builder 


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1].content
    if "Please proceed" in last_message:
        return "continue" 
    else:
        return "redo"
    
    

class HumanApprovalNode:
    def __init__(self, name: str, stop_count: int = 5, 
                get_action: Callable = lambda: input("Do you like this paper? (y/n/back): ").lower(), 
                get_reject_feedback: Callable = lambda: input("Please provide feedback for rejection: "),
                print_paper: Callable = lambda paper: print(f"Title: {paper['title']}\nScore: {paper['score']}\nAI Comment: {paper['comment']}"),
                print_action_error: Callable = lambda: print("Invalid action. Please choose y, n, or back."),
                print_summary_begin: Callable = lambda: print("--- Failed to recommend papers, reflecting on human preference ---"),
                print_summary: Callable = lambda summary: print(f"Summary of human preference: \n{summary}")):
        
        self.name = name
        self.stop_count = stop_count
        self.get_action = get_action
        self.get_reject_feedback = get_reject_feedback
        self.print_paper = print_paper
        self.print_action_error = print_action_error
        self.print_summary_begin = print_summary_begin
        self.print_summary = print_summary
        
    def __call__(self, state: State):
        """ 
        Simple Version | We do a top-down approach, wait until human approves 5 papers before we proceed
        """
        possible_actions = ["y", "n", "back"]
        
        papers = state["papers"]
        
        papers.sort(key=lambda x: x['score'], reverse=True)
        
        print("Presenting Top Papers selected by AI")
        approved_count = 0
        for paper in papers:
            self.print_paper(paper)
            
            if approved_count >= self.stop_count:
                continue
             
            while True:
                action = self.get_action()
                if action in possible_actions:
                    break
                self.print_action_error()
            
            if action == "y":
                paper['score'] = 999  # Indicate human approval
                approved_count += 1
            elif action == "n":
                argument = self.get_reject_feedback()
                paper['argument'] = argument
                paper["score"] = 0
            elif action == "back":
                self.print_summary_begin()
                infered_preference = summarize_human_preference(papers, model)
                self.print_summary(infered_preference)
                
                # AI should do analysis on human preference and do the ranking again
                return {"messages": [{"role": "assistant", "content": "Please redo the paper selection process."},
                                     {"role": "assistant", "content": f"Summary of human preference: {infered_preference}"}], 
                        "papers": papers}
        
        return {"messages": [{"role": "assistant", "content": "Please proceed to the next step."}], "papers": papers}
    
def build_research_graph(
    get_action: Callable = lambda: input("Do you like this paper? (y/n/back): ").lower(), 
    get_reject_feedback: Callable = lambda: input("Please provide feedback for rejection: "),
    print_paper: Callable = lambda paper: print(f"Title: {paper['title']}\nScore: {paper['score']}\nAI Comment: {paper['comment']}"),
    print_action_error: Callable = lambda: print("Invalid action. Please choose y, n, or back."),
    print_summary_begin: Callable = lambda: print("--- Failed to recommend papers, reflecting on human preference ---"),
    print_summary: Callable = lambda summary: print(f"Summary of human preference: \n{summary}")):
    """ 
    1. Scrape ArXiv for newest 200 AI papers 
    2. AI do ranking & analysis base on abstracts
    3. Human select 5 papers from top-ranked papers
    4. Conditional on human acceptance, proceed, otherwise, summarize human preference and go back to 2.
    -- Output 5 papers & analysis
    """
    model = ChatOpenAI(model="gpt-4o")
    builder = StateGraph(State)
    
    # Research graph part
    research_entry_point = "crawl_arxiv_node"
    builder.add_node(research_entry_point, CrawlNode(research_entry_point))
    builder.add_edge(START, research_entry_point)
    
    builder = add_selection_nodes(builder, research_entry_point, "human_approval_node", model)
    
    # Human approval graph part
    human_approval_node = "human_approval_node"
    builder.add_node(human_approval_node, HumanApprovalNode(human_approval_node,
                                                            get_action=get_action,
                                                            get_reject_feedback=get_reject_feedback,
                                                            print_paper=print_paper,
                                                            print_action_error=print_action_error,
                                                            print_summary_begin=print_summary_begin,
                                                            print_summary=print_summary
                                                            ))  # Assuming HumanApprovalNode is defined elsewhere
    
    # So here we ought to have a conditional edge, if there is "redo" in the message, we re-run from research_entry_point
    builder.add_conditional_edges(human_approval_node, 
                                  should_continue,
                                  {
                                    "continue": END,
                                    "redo": research_entry_point,
                                  },)
        
    return builder.compile()