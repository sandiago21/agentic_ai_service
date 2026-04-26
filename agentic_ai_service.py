from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel

import requests
import re
import warnings

warnings.filterwarnings("ignore")
import json
import logging
from typing import TypedDict, Annotated, Dict
from json_repair import repair_json
from bs4 import BeautifulSoup
from pydantic import Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
# from langchain.agents import create_tool_calling_agent


api = FastAPI()


# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)


class Config(object):
    def __init__(self):
        self.random_state = 42
        self.max_len = 256
        self.reasoning_max_len = 256
        self.temperature = 0.01
        self.repetition_penalty = 1.2
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        # self.reasoning_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        # self.reasoning_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        # self.reasoning_model_name = "Qwen/Qwen2.5-7B-Instruct"


config = Config()


tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=torch.float16, device_map=config.DEVICE
)

# reasoning_tokenizer = AutoTokenizer.from_pretrained(config.reasoning_model_name)
# reasoning_model = AutoModelForCausalLM.from_pretrained(
#     config.reasoning_model_name,
#     torch_dtype=torch.float16,
#     device_map=config.DEVICE
# )


def generate(prompt):
    """
    Generate a text completion from a causal language model given a prompt.
    Parameters
    ----------
    prompt : str
        Input text prompt used to condition the language model.
    Returns
    -------
    str
        The generated continuation text, decoded into a string with special
        tokens removed and leading/trailing whitespace stripped.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_len,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def reasoning_generate(prompt):
    """
    Generate a text completion from a causal language model given a prompt.
    Parameters
    ----------
    prompt : str
        Input text prompt used to condition the language model.
    Returns
    -------
    str
        The generated continuation text, decoded into a string with special
        tokens removed and leading/trailing whitespace stripped.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.reasoning_max_len,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


class Action(BaseModel):
    tool: str = Field(...)
    args: Dict


# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    proposed_action: str
    information: str
    raw_output: str
    output: str
    confidence: float
    judge_explanation: str


ALL_TOOLS = {
    "web_search": ["query"],
    "visit_webpage": ["url"],
}

ALLOWED_TOOLS = {
    "web_search": ["query"],
    "visit_webpage": ["url"],
}


def visit_webpage(url: str) -> str:
    """
    Fetch and read the content of a webpage.
    Args:
        url: URL of the webpage
    Returns:
        Extracted readable text (truncated)
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = [p.get_text() for p in soup.find_all("p")]
    text = "\n".join(paragraphs)

    return (text[:500], text[500:1000])


def visit_webpage(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()

    # Extract more elements (not just <p>)
    elements = soup.find_all(["p", "dd"])

    text = " \n ".join(el.get_text(strip=False) for el in elements)

    return (text[:1000],)


def visit_webpage(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()

    content = soup.find("div", {"id": "mw-content-text"})

    texts = []

    # 1. Paragraphs
    for p in content.find_all("p"):
        texts.append(p.get_text(strip=False))

    # 2. Definition lists
    for dd in content.find_all("dd"):
        texts.append(dd.get_text(strip=False))

    # 3. Tables (IMPORTANT)
    for table in content.find_all("table", {"class": "wikitable"}):
        for row in table.find_all("tr"):
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if cols:
                texts.append(" | ".join(cols))

    return (" \n ".join(texts)[:1000],)


def visit_webpage(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()

    content = soup.find("div", {"id": "mw-content-text"})

    # Extract more elements (not just <p>)
    elements = soup.find_all(["p", "dd"])

    main_text = " \n ".join(el.get_text(strip=False) for el in elements)

    # 3. Tables (IMPORTANT)
    table_texts = []
    for table in content.find_all("table", {"class": "wikitable"}):
        for row in table.find_all("tr"):
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if cols:
                table_texts.append(" | ".join(cols))

    if len(table_texts) > 0:
        return [
            main_text[:1000],
            " \n ".join(table_texts),
        ]
    else:
        return [
            main_text[:1000],
        ]


def visit_webpage(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()

    content = soup.find("div", {"id": "mw-content-text"})

    # Extract more elements (not just <p>)
    elements = soup.find_all(["p", "dd"])

    main_text = " \n ".join(el.get_text(strip=False) for el in elements)

    # 3. Tables (IMPORTANT)
    table_texts = []
    if content is not None:
        for table in content.find_all("table", {"class": "wikitable"}):
            for row in table.find_all("tr"):
                cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if cols:
                    table_texts.append(" | ".join(cols))

    if len(table_texts) > 0:
        return [
            main_text[:1000],
            " \n ".join(table_texts),
        ]
    else:
        return [
            main_text[:1000],
        ]


def visit_webpage_wiki(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()

    content = soup.find("div", {"id": "mw-content-text"})

    # Extract more elements (not just <p>)
    elements = soup.find_all(["p", "dd"])

    main_text = " \n ".join(el.get_text(strip=False) for el in elements)

    # 3. Tables (IMPORTANT)
    table_texts = []
    if content is not None:
        for table in content.find_all("table", {"class": "wikitable"}):
            for row in table.find_all("tr"):
                cols = [c.get_text(strip=False) for c in row.find_all(["td", "th"])]
                if cols:
                    table_texts.append(" | ".join(cols))

    if len(table_texts) > 0:
        return [
            main_text[:1000],
            " \n ".join(table_texts)[:5000],
        ]
    else:
        return [
            main_text[:1000],
        ]


def visit_webpage_main(url: str):
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()

    # 🔥 Try to focus on body (fallback if no clear container)
    content = soup.find("body")

    # ✅ Extract broader set of elements
    elements = content.find_all(["p", "dd", "td", "div"])

    texts = []
    for el in elements:
        text = el.get_text(strip=True)
        if text and len(text) > 30:  # filter noise
            texts.append(text)

    main_text = "\n".join(texts)

    # ✅ Extract all tables (not just wikitable)
    table_texts = []
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if cols:
                table_texts.append(" | ".join(cols))

    if table_texts:
        return [main_text[:1500], "\n".join(table_texts)[:5000]]
    else:
        return [main_text[:1500]]


def web_search(query: str, num_results: int = 10):
    """
    Search the internet for the query provided
    Args:
        query: Query to search in the internet
    Returns:
        list of urls
    """

    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.post(url, data={"q": query}, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")
    return [a.get("href") for a in soup.select(".result__a")[:num_results]]


def planner_node(state: AgentState):
    """
    Planning node for a tool-using LLM agent.
    The planner enforces:
    - Strict JSON-only output
    - Tool selection constrained to predefined tools
    - Argument generation limited to user-provided information
    Parameters
    ----------
    state : dict
        Agent state dictionary containing:
        - "messages" (str): The user's natural language request.
    Returns
    -------
    dict
        Updated state dictionary with additional keys:
        - "proposed_action" (dict): Parsed JSON tool call in the form:
              {
                  "tool": "<tool_name>",
                  "args": {...}
              }
        - "risk_score" (float): Initialized risk score (default 0.0).
        - "decision" (str): Initial decision ("allow" by default).
    Behavior
    --------
    1. Constructs a planning prompt including:
       - Available tools and allowed arguments
       - Strict JSON formatting requirements
       - Example of valid output
    2. Calls the language model via `generate()`.
    3. Attempts to extract valid JSON from the model output.
    4. Repairs malformed JSON using `repair_json`.
    5. Stores the parsed action into the agent state.
    Security Notes
    --------------
    - This node does not enforce tool-level authorization.
    - It does not validate hallucinated tools.
    - It does not perform risk scoring beyond initializing values.
    - Downstream nodes must implement:
        * Tool whitelist validation
        * Argument validation
        * Risk scoring and mitigation
        * Execution authorization
    Intended Usage
    --------------
    Designed for multi-agent or LangGraph-style workflows where:
        Planner → Risk Assessment → Tool Executor → Logger
    This node represents the *planning layer* of the agent architecture.
    """

    user_input = state["messages"][-1].content

    prompt = f"""
You are a planning agent.
You MUST return ONLY valid JSON as per the tools specs below ONLY.
No extra text.
DO NOT invent anything additional beyond the user request provided. Keep it strict to the user request information provided. The question and the query should be fully relevant to the user request provided, no deviation and hallucination. If possible and makes sense then the query should be exactly the user request.
The available tools and their respective arguments are: {{
    "web_search": ["query"],
    "visit_webpage": ["url"],
}}
Return exactly the following format:
Response:
{{
  "tool": "...",
  "args": {{...}}
}}
User request: Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?. Example of valid JSON expected:
Response:
{{"tool": "web_search",
 "args": {{"query": "Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?",
  }}
}}
Return only one Response!
User request:
{user_input}
"""

    output = generate(prompt)

    state["proposed_action"] = output.split("Response:")[-1]
    fixed = repair_json(state["proposed_action"])
    data = json.loads(fixed)
    state["proposed_action"] = data

    return state


def safety_node(state: AgentState):
    """
    Evaluate the information provided and output the response for the user request.
    """

    user_input = state["messages"][-1].content
    information = state["information"]

    prompt = f"""
You are a reasoning agent who takes into account the provided information, if any, and answers to the user request.
You must reason over the user request and the provided information and output the answer to the user's request. Reason well and thoroughly over the information provided, if any, and output the answer to the user's question exactly.
You MUST ONLY return EXACTLY the answer to the user's question in the following format:
Response: <answer>
DO NOT add anything additional and return ONLY what is asked and in the format asked.
If you output anything else, it is incorrect.
If there is no information provided or the information is not relevant then answer as best based on your own knowledge.
Example of valid json response for user request: Who was the winner of 2025 World Snooker Championship:
Response: Zhao Xintong.
Example of valid json response for user request: What is the first name of the winner of 2025 World Snooker Championship:
Response: Zhao.
User request:
{user_input}
Information:
{information}
"""

    raw_output = reasoning_generate(prompt)
    # raw_output = generate(prompt)

    logger.info(f"Raw Output: {raw_output}")

    raw = raw_output.strip()

    matches = re.findall(r"Response:\s*([^\n]+)", raw)

    if matches:
        output = matches[-1].strip()  # ✅ take LAST occurrence
    else:
        # Find the first valid "Response: ..." occurrence
        match = re.search(r"Response:\s*([^\n\.]+)", raw)

        if match:
            output = match.group(1).strip()
        else:
            # fallback: take first line
            output = raw.split("\n")[0].strip()

        if "Response:" in output:
            output = output.split("Response:")[-1]
        elif "Response" in output:
            output = output.split("Response")[-1]

        # Clean quotes / trailing punctuation
        output = output.strip('"').strip()
        if output.endswith("."):
            output = output[:-1]

    # Clean
    if "Response:" in output:
        output = output.split("Response:")[-1]
    elif "Response" in output:
        output = output.split("Response")[-1]

    # Clean quotes / trailing punctuation
    output = output.strip('"').strip()
    if output.endswith("."):
        output = output[:-1]

    if output == "":
        # Find the first valid "Response: ..." occurrence
        match = re.search(r"Response:\s*([^\n\.]+)", raw)

        if match:
            output = match.group(1).strip()
        else:
            # fallback: take first line
            output = raw.split("\n")[0].strip()

        if "Response:" in output:
            output = output.split("Response:")[-1]
        elif "Response" in output:
            output = output.split("Response")[-1]

        # Clean quotes / trailing punctuation
        output = output.strip('"').strip()
        if output.endswith("."):
            output = output[:-1]

    output = output.split(".")[0]

    output = output.split("|")[0]

    state["output"] = output.strip()

    logger.info(f"State (Safety Agent): {state}")

    return state


def Judge(state: AgentState):
    """
    Evaluate whether the answer provided is indeed based on the information provided or not.
    """

    answer = state["output"]
    information = state["information"]
    user_input = state["messages"][-1].content

    prompt = f"""
You are a Judging agent.
You must reason over the user request and judge with a confidence score whether the answer is indeed based on the provided information or not. 
Example: User request: Who was the winner of 2025 World Snooker Championship?
Information: Zhao Xintong won the 2025 World Snooker Championship with a dominant 18-12 final victory over Mark Williams in Sheffield on Monday. The 28 year-old becomes the first player from China to win snooker’s premier prize at the Crucible Theatre.
Zhao, who collects a top prize worth £500,000, additionally becomes the first player under amateur status to go all the way to victory in a World Snooker Championship.
The former UK champion entered the competition in the very first qualifying round at the English Institute of Sport last month.
He compiled a dozen century breaks as he fought his way through four preliminary rounds in fantastic fashion to qualify for the Crucible for the third time in his career.
In the final round of the qualifiers known as Judgement Day, Zhao edged Elliot Slessor 10-8 in a high-quality affair during which both players made a hat-trick of tons.
Ironically, that probably represented his sternest test throughout the entire event.
Answer: "Zhao Xintong"
Response: {{
    "confidence": 1.0,
    "explanation": Based on the information provided, it is indeed mentioned that Zhao Xingong, which is the answer provided, won the 2025 World Snooker Championship.
}}
Example: User request: Who was the winner of 2025 World Snooker Championship?
Information: Zhao Xintong won the 2025 World Snooker Championship with a dominant 18-12 final victory over Mark Williams in Sheffield on Monday. The 28 year-old becomes the first player from China to win snooker’s premier prize at the Crucible Theatre.
Zhao, who collects a top prize worth £500,000, additionally becomes the first player under amateur status to go all the way to victory in a World Snooker Championship.
The former UK champion entered the competition in the very first qualifying round at the English Institute of Sport last month.
He compiled a dozen century breaks as he fought his way through four preliminary rounds in fantastic fashion to qualify for the Crucible for the third time in his career.
In the final round of the qualifiers known as Judgement Day, Zhao edged Elliot Slessor 10-8 in a high-quality affair during which both players made a hat-trick of tons.
Ironically, that probably represented his sternest test throughout the entire event.
Answer: "Ronnie O'sullivan"
Response: {{
    "confidence": 0.0,
    "explanation": Based on the information provided, it is was Zhao Xingong and not Ronnie O'sullivan who won the 2025 World Snooker Championship.
}}
Example: User request: Who was the winner of 2025 World Snooker Championship?
Information:  
Answer: "Ronnie O'sullivan"
Response: {{
    "confidence": 0.0,
    "explanation": There is no information provided, so cannot answer who won the 2025 World Snooker Championship.
}}
Return exactly the above requested format and nothing more! 
DO NOT generate any additional text after it! 
Return only what is asked and in the format asked!
User request:
{user_input}
Information:
{information}
Answer:
{answer}
"""

    raw_output = generate(prompt)

    print(f"Judge raw output: {raw_output}")

    output = raw_output.split("Response:")[-1].strip()
    fixed = repair_json(output)
    data = json.loads(fixed)

    state["confidence"] = data["confidence"]
    state["judge_explanation"] = data["explanation"]

    # logger.info(f"State (Judge Agent): {state}")

    return state


def route(state: AgentState):
    """Determine the next step based on Safety Agent classification"""
    if state["risk_score"] > 0.5:
        return "block"
    else:
        return "allow"


def tool_executor(state: AgentState):
    """
    Tool execution node for a risk-aware LLM agent.
    This node executes the validated and approved tool call proposed by the
    planner and assessed by the safety layer. It conditionally dispatches
    execution based on the safety decision and updates the agent state with
    the final output.
    Parameters
    ----------
    state : dict
        Agent state dictionary containing:
        - "decision" (str): Safety decision ("allow" or blocking variant).
        - "risk_score" (float): Computed risk score.
        - "proposed_action" (dict): Validated tool call in structured form.
    Returns
    -------
    dict
        Updated state dictionary including:
        - "output" (str): Result of tool execution OR block message.
    Execution Flow
    --------------
    1. If the safety decision is not "allow":
       - Skip tool execution.
       - Return a blocked message including the risk score.
    2. If allowed:
       - Validate the proposed action using the `Action` schema.
       - Dispatch execution to the appropriate tool implementation:
            * "google_calendar"
            * "reply_email"
            * "share_credentials"
       - Store tool result in `state["output"]`.
    3. If the tool is unrecognized:
       - Return "Unknown tool" as a fallback response.
    Security Considerations
    -----------------------
    - Execution only occurs after passing the safety node.
    - No runtime sandboxing is implemented.
    - No per-tool authorization layer (RBAC) is enforced.
    - Sensitive tools (e.g., credential exposure) should require:
        * Elevated approval thresholds
        * Human-in-the-loop confirmation
        * Additional auditing
    Architectural Role
    ------------------
    Planner → Safety → Tool Execution → Logger
    This node represents the controlled execution layer of the agent,
    responsible for translating structured LLM intent into real system actions.
    """

    try:
        webpage_result = ""
        action = Action.model_validate(state["proposed_action"])

        best_query_webpage_information_similarity_score = -1.0
        best_webpage_information = ""

        webpage_information_complete = ""

        if action.tool == "web_search":
            logger.info(f"action.tool: {action.tool}")

            query_embeddings = sentence_transformer_model.encode_query(
                state["messages"][-1].content
            ).reshape(1, -1)
            query_arg_embeddings = sentence_transformer_model.encode_query(
                state["proposed_action"]["args"]["query"]
            ).reshape(1, -1)
            score = float(
                cosine_similarity(query_embeddings, query_arg_embeddings)[0][0]
            )

            if score > 0.80:
                results = web_search(**action.args)
            else:
                logger.info(
                    f"Overwriting user query because the Agent suggested query had score: {state['proposed_action']['args']['query']} - {score}"
                )
                results = web_search(**{"query": state["messages"][-1].content})

            logger.info(f"Webpages - Results: {results}")

            for result in results:
                try:
                    webpage_results = visit_webpage_wiki(result)
                    webpage_result = " \n ".join(webpage_results)

                    # for webpage_result in webpage_results:
                    query_embeddings = sentence_transformer_model.encode_query(
                        state["messages"][-1].content
                    ).reshape(1, -1)
                    webpage_information_embeddings = (
                        sentence_transformer_model.encode_query(webpage_result).reshape(
                            1, -1
                        )
                    )
                    query_webpage_information_similarity_score = float(
                        cosine_similarity(
                            query_embeddings, webpage_information_embeddings
                        )[0][0]
                    )

                    # logger.info(f"Webpage Information and Similarity Score: {result} - {webpage_result} - {query_webpage_information_similarity_score}")

                    if query_webpage_information_similarity_score > 0.65:
                        webpage_information_complete += webpage_result
                        webpage_information_complete += " \n "
                        webpage_information_complete += " \n "

                    if (
                        query_webpage_information_similarity_score
                        > best_query_webpage_information_similarity_score
                    ):
                        best_query_webpage_information_similarity_score = (
                            query_webpage_information_similarity_score
                        )
                        best_webpage_information = webpage_result

                    webpage_results = visit_webpage_main(result)
                    webpage_result = " \n ".join(webpage_results)

                    # for webpage_result in webpage_results:
                    query_embeddings = sentence_transformer_model.encode_query(
                        state["messages"][-1].content
                    ).reshape(1, -1)
                    webpage_information_embeddings = (
                        sentence_transformer_model.encode_query(webpage_result).reshape(
                            1, -1
                        )
                    )
                    query_webpage_information_similarity_score = float(
                        cosine_similarity(
                            query_embeddings, webpage_information_embeddings
                        )[0][0]
                    )

                    # logger.info(f"Webpage Information and Similarity Score: {result} - {webpage_result} - {query_webpage_information_similarity_score}")

                    if query_webpage_information_similarity_score > 0.65:
                        webpage_information_complete += webpage_result
                        webpage_information_complete += " \n "
                        webpage_information_complete += " \n "

                    if (
                        query_webpage_information_similarity_score
                        > best_query_webpage_information_similarity_score
                    ):
                        best_query_webpage_information_similarity_score = (
                            query_webpage_information_similarity_score
                        )
                        best_webpage_information = webpage_result

                except Exception as e:
                    logger.info(f"Tool Executor - Exception: {e}")

        elif action.tool == "visit_webpage":
            try:
                if "www.youtube.com" in str(action.args["url"]):
                    video_id = action.args["url"].split("www.youtube.com/watch?v=")[-1]
                    api = YouTubeTranscriptApi()
                    transcript = api.fetch(video_id)
                    texts = [x.text for x in transcript]
                    webpage_information_complete = " \n ".join(texts)

                    index = 0
                    counter = 0
                    best_query_webpage_information_similarity_score = 0.0
                    for text in texts:
                        query_embeddings = sentence_transformer_model.encode_query(
                            state["messages"][-1].content
                        ).reshape(1, -1)
                        webpage_information_embeddings = (
                            sentence_transformer_model.encode_query(text).reshape(1, -1)
                        )
                        query_webpage_information_similarity_score = float(
                            cosine_similarity(
                                query_embeddings, webpage_information_embeddings
                            )[0][0]
                        )

                        if (
                            query_webpage_information_similarity_score
                            > best_query_webpage_information_similarity_score
                        ):
                            best_query_webpage_information_similarity_score = (
                                query_webpage_information_similarity_score
                            )
                            index = counter

                        counter += 1

                    webpage_information_complete = f"""answer: {texts[index + 1]}"""
                    state["best_query_webpage_information_similarity_score"] = 1.0

                else:
                    webpage_results = visit_webpage_wiki(action.args["url"])
                    webpage_result = " \n ".join(webpage_results)

                    # for webpage_result in webpage_results:
                    query_embeddings = sentence_transformer_model.encode_query(
                        state["messages"][-1].content
                    ).reshape(1, -1)
                    webpage_information_embeddings = (
                        sentence_transformer_model.encode_query(webpage_result).reshape(
                            1, -1
                        )
                    )
                    query_webpage_information_similarity_score = float(
                        cosine_similarity(
                            query_embeddings, webpage_information_embeddings
                        )[0][0]
                    )

                    # logger.info(f"Webpage Information and Similarity Score: {result} - {webpage_result} - {query_webpage_information_similarity_score}")

                    if query_webpage_information_similarity_score > 0.65:
                        webpage_information_complete += webpage_result
                        webpage_information_complete += " \n "
                        webpage_information_complete += " \n "

                    if (
                        query_webpage_information_similarity_score
                        > best_query_webpage_information_similarity_score
                    ):
                        best_query_webpage_information_similarity_score = (
                            query_webpage_information_similarity_score
                        )
                        best_webpage_information = webpage_result

                    webpage_results = visit_webpage_main(action.args["url"])
                    webpage_result = " \n ".join(webpage_results)

                    # for webpage_result in webpage_results:
                    query_embeddings = sentence_transformer_model.encode_query(
                        state["messages"][-1].content
                    ).reshape(1, -1)
                    webpage_information_embeddings = (
                        sentence_transformer_model.encode_query(webpage_result).reshape(
                            1, -1
                        )
                    )
                    query_webpage_information_similarity_score = float(
                        cosine_similarity(
                            query_embeddings, webpage_information_embeddings
                        )[0][0]
                    )

                    # logger.info(f"Webpage Information and Similarity Score: {result} - {webpage_result} - {query_webpage_information_similarity_score}")

                    if query_webpage_information_similarity_score > 0.65:
                        webpage_information_complete += webpage_result
                        webpage_information_complete += " \n "
                        webpage_information_complete += " \n "

                    if (
                        query_webpage_information_similarity_score
                        > best_query_webpage_information_similarity_score
                    ):
                        best_query_webpage_information_similarity_score = (
                            query_webpage_information_similarity_score
                        )
                        best_webpage_information = webpage_result
            except Exception as e:
                webpage_information_complete = str(e)
                pass
        elif "answer" in state["proposed_action"]:
            webpage_information_complete = (
                f"""answer: {state["proposed_action"]["answer"]}"""
            )
            state["best_query_webpage_information_similarity_score"] = 1.0
        else:
            webpage_information_complete = ""

        if (
            webpage_information_complete == ""
            and best_query_webpage_information_similarity_score > 0.30
        ):
            webpage_information_complete = best_webpage_information

        state["information"] = webpage_information_complete[:3000]
        state["best_query_webpage_information_similarity_score"] = (
            best_query_webpage_information_similarity_score
        )
    except:
        if "answer" in state["proposed_action"]:
            webpage_information_complete = (
                f"""answer: {state["proposed_action"]["answer"]}"""
            )
            state["information"] = webpage_information_complete
            state["best_query_webpage_information_similarity_score"] = 1.0
        else:
            state["information"] = ""
            state["best_query_webpage_information_similarity_score"] = -1.0

    # logger.info(f"Information: {state['information']}")
    # logger.info(f"Information: {state['best_query_webpage_information_similarity_score']}")

    return state


safe_workflow = StateGraph(AgentState)
# safe_workflow = StateGraph(dict)

safe_workflow.add_node("planner", planner_node)
safe_workflow.add_node("tool_executor", tool_executor)
safe_workflow.add_node("safety", safety_node)
# safe_workflow.add_node("judge", Judge)

# safe_workflow.set_entry_point("planner")

safe_workflow.add_edge(START, "planner")
safe_workflow.add_edge("planner", "tool_executor")
safe_workflow.add_edge("tool_executor", "safety")
# safe_workflow.add_edge("safety", "judge")
# safe_workflow.add_conditional_edges(
#     "safety",
#     route,
#     {
#         "allow": "tool_executor",
#         "block": END,
#     },
# )
# safe_workflow.add_edge("tool_executor", END)

# safe_app = safe_workflow.compile()


# --- Agent Definition ---
class Agent:
    def __init__(self):
        self.safe_app = safe_workflow.compile()
        print("Agent initialized.")

    def __call__(self, question: str, filename: str) -> str:
        state = {
            "messages": question,
        }

        if len(tokenizer.encode(state["messages"][::-1])) < len(
            tokenizer.encode(state["messages"])
        ):
            state["messages"] = state["messages"][::-1]

        try:
            response = self.safe_app.invoke(state)

            if "answer: " in response["information"]:
                response["output"] = (
                    response["information"].split("answer: ")[-1].strip()
                )

            agent_answer = response["output"]

        except Exception as e:
            agent_answer = str(e)

        return agent_answer


all_questions_and_answers = []
agent = Agent()


class Query(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description="User's question to be answered by the agent",
    )
    filename: Optional[str] = Field(
        None,
        min_length=3,
        max_length=512,
        description="Name of file containing useful information to be used by the agent",
    )
    answer: Optional[str] = Field(
        None,
        min_length=3,
        max_length=512,
        description="Answer provided by the agent to the user's question",
    )


@api.get("/queries", response_model=List[Query])
def get_queries(first_n: int = None):
    if first_n:
        return all_questions_and_answers[:first_n]

    return all_questions_and_answers


@api.post("/ask_question")
def get_answer_to_question(query: Query):
    for question_and_answer in all_questions_and_answers:
        if query.question == question_and_answer.question:
            return question_and_answer.answer

    agent_answer = agent.__call__(query.question, filename=query.filename)
    query.answer = agent_answer

    all_questions_and_answers.append(query)

    return query.answer
