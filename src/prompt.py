EXTRACT_CITATIONS_AND_TAGS_PROMPT = """
Analyze the following academic paper text and extract:
1. A list of citations (titles of cited papers)
2. A list of relevant tags or keywords

Provide your answer in JSON format with two keys: "citations" and "tags".
Each should be a list of strings. For example:
{
    "citations": ["Joint 2d-3d-semantic data for indoor scene understanding", "Scenescript: Reconstruct-ing scenes with an autoregressive structured language model", ...],
    "tags": ["machine learning", "computer vision", ...]
}

Here's the paper text:

"""

EXTRACT_CITATIONS_PROMPT = """
Analyze the following academic paper text and extract a list of citations (titles of cited papers).

Provide your answer in JSON format with a single key: "citations".
It should be a list of strings. For example:
{
    "citations": ["Joint 2d-3d-semantic data for indoor scene understanding", "Scenescript: Reconstructing scenes with an autoregressive structured language model", ...]
}

Here's the paper text:

"""


EXTRACT_TAGS_PROMPT = """
Analyze the following academic paper text and extract a list of relevant tags or keywords.

Provide your answer in JSON format with a single key: "tags".
It should be a list of strings. For example:
{
    "tags": ["machine learning", "computer vision", ...]
}

Here's the paper text:

"""