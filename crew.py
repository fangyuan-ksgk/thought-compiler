from crewai_tools import SeleniumScrapingTool
from crewai import Agent, Task, Crew, Process
import os 

os.environ["SERPER_API_KEY"] = "5affe20ad27423ecb9954ffa3151680c42adadca"
os.environ["SERPLY_API_KEY"] = "Xr5jNnnGynMUgAWqEC1vtyCB"

scrape_tool = SeleniumScrapingTool()

useful_links = [
  "https://huggingface.co/papers?date=2024-08-08",
  "https://arxiv.org/list/cs/new"
]

researcher = Agent(
  role="Researcher",
  goal=f"""Research the selected paper and suggest five interesting perspectives that could form the basis for the article. Look through provided links to find interesting papers and provide a summary of the main points and the paper.
  Useful links: {useful_links[0]}""",
  backstory="""You are a Research Scientist at Google DeepMind focusing on AI.
  You are good at using the provided link to scrape the paper and summarise the main points, as well as search the web for related introduction to get a better perspectives on how to understand the topic.""",
  verbose=True,
  allow_delegation=True,
  tools=[scrape_tool]
)

task1 = Task(
  description="""Look through the new AI papers inside the provided link and suggest 5 most interesting papers with their link andabstracts.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

# Instantiate your crew with a sequential process
crew = Crew(
  # agents=[finder, researcher, writer],
  agents = [researcher],
  # tasks=[task1, task2, task3],
  tasks = [task1],
  verbose=True,
  process = Process.sequential
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)