# Structured State 
import dataclasses
from dataclasses import dataclass
from typing import Callable
import json
from .model import route_model, Engine


# 1st version focus on the article write-ups 
@dataclass
class IdeaPrompt:
    context: str
    feedback: str = ""
    preferred: str = ""

    def get_prompt(self) -> str:
        additions = []
        if self.preferred:
            additions.append(f"You need to provide your ideas. For reference, here are some of the preferred ideas:\n{self.preferred}")
        if self.feedback:
            additions.append(f"And human feedback:\n{self.feedback}")
        
        base = f"Given context:\n{self.context}\n"
        
        if additions:
            base += ' '.join(additions) + "\n"
        
        base += """Provide 5 Ideas in JSON format:
{
    "idea1": "",
    "idea2": "",
    "idea3": "",
    "idea4": "",
    "idea5": ""
}"""
        return base
    
    @classmethod
    def format(cls, context: str, feedback: str = "", preferred: str = "") -> str:
        return cls(context, feedback, preferred).get_prompt()
    

# def get_response(prompt: str) -> str:
#     raise NotImplementedError


def parse_ideas(response):
    """ 
    Identify { / } pattern to parse out ideas enumeration
    """
    if isinstance(response, str):
        txt = response 
    else:
        txt = response[0].text 
    
    import re
    import json
    
    # Find all occurrences of JSON-like structures
    json_patterns = re.findall(r'\{[^{}]*\}', txt)
    
    ideas = []
    for pattern in json_patterns:
        try:
            # Attempt to parse each pattern as JSON
            idea = json.loads(pattern)
            for (k,v) in idea.items():
                ideas.append(v)
        except json.JSONDecodeError:
            # If parsing fails, skip this pattern
            continue
    
    return ideas



# An Idea is born under some context, it will have multiple options, human feedback is provided to like / dislike and edit them
@dataclass
class Idea: # you always have choices to branch out on some ideas
    options: list  
    preferences: list
    feedback: str
    context: str
    model: Engine

    @classmethod
    def make(cls, context: str, model_name: str = "Claude"):
        model = route_model(model_name)
        response = model.get_response(IdeaPrompt.format(context=context)) # generate content
        options = parse_ideas(response)
        return cls(options, [False]*len(options), "", context, model)
    
    @property
    def content(self): # pick the favorite idea from the liked options
        if any(self.preferences):
            return self.options[self.preferences.index(True)]
        else:
            return self.options[-1]
        
    def save(self, filename: str = "cave/idea.json"):
        data = {
            "options": self.options,
            "preferences": self.preferences,
            "feedback": self.feedback,
            "context": self.context,
            "model_name": self.model.__class__.__name__
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename: str = "cave/idea.json"):
        with open(filename, 'r') as f:
            data = json.load(f)
        model = route_model(data['model_name'])
        idea = cls(
            options=data['options'],
            preferences=data['preferences'],
            feedback=data['feedback'],
            context=data['context'],
            model=model
        )
        return idea
        
    def regenerate(self): # think again
        preferred = "\n".join([f"{i+1}. {option}" for i, option in enumerate(self.options) if self.preferences[i]])
        response = self.model.get_response(IdeaPrompt.format(context=self.context, feedback=self.feedback, preferred=preferred))
        self.options = parse_ideas(response)
        self.preferences = [False]*len(self.options)

    def like(self, choice):
        self.preferences[choice] = True
    
    def dislike(self, choice):
        self.preferences[choice] = False
    
    def edit(self, choice, edited_content):
        self.options[choice] = edited_content

    def add_feedback(self, new_feedback):
        self.feedback = new_feedback