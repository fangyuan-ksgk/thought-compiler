# Plan object chain-up several points where Idea is required

import dataclasses
from dataclasses import dataclass
from typing import Callable
from typing import List
from .tool import Tool
from .idea import Idea

# It's rather simple, at each step, there 

@dataclass 
class Goal:
    name: str
    description: str
    status: bool = False

@dataclass
class State: # Single most important object of the entire codebase right here
    idea: Idea
    # tools: List[Tool] # Tools available | Rid of the tools for now | Just chat it out
    goals: List[Goal] # Goals to achieve
   
@dataclass
class Plan:
    states: List[State]
    current_step: int = 0

    @property
    def current_state(self):
        return self.states[self.current_step]

    def load(self, filename):
        raise NotImplementedError
    
    def save(self, filename):
        raise NotImplementedError
    
    def next(self):
        self.current_step += 1

    def prev(self):
        self.current_step -= 1

    def regenerate(self):
        self.current_state.idea.regenerate()

    def like(self, choice):
        self.current_state.idea.like(choice)

    def dislike(self, choice):
        self.current_state.idea.dislike(choice)

    def edit(self, choice, edited_content):
        self.current_state.idea.edit(choice, edited_content)