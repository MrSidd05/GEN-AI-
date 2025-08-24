#PRODUCT REVIEW ANALYZER
## Starter code: provide your solutions in the TODO parts
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# First initialize your LLM
model_id = "meta-llama/llama-3-3-70b-instruct" ## Or you can use other LLMs available via watsonx.ai

# Use these parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 512,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.2, # this randomness or creativity of the model's responses
}

url = "https://us-south.ml.cloud.ibm.com"
project_id = "skills-network"

# TODO: Initialize your LLM
llm = WatsonxLLM(
    model_id = model_id,
    project_id = project_id,
    url = url,
    params = parameters)

# Here is an example template you can use
template = """
Analyze the following product review:
"{review}"

Provide your analysis in the following format:
- Sentiment: (positive, negative, or neutral)
- Key Features Mentioned: (list the product features mentioned)
- Summary: (one-sentence summary)
"""

# TODO: Create your prompt template
product_review_prompt = PromptTemplate.from_template(template)

# TODO: Create a formatting function
def format_review_prompt(variables):
    return product_review_prompt.format(**variables)
    pass

# TODO: Build your LCEL chain
review_analysis_chain = (
    RunnableLambda(format_review_prompt)
    | llm
    | StrOutputParser()
)

# Example reviews to process
reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped working after just two months. Customer service was unhelpful."
]

# TODO: Process the reviews
for i,review in enumerate(reviews):
   print(f"=== Review #{i+1} ===")
   result = review_analysis_chain.invoke({"review": reviews})
   print(result)
   print()