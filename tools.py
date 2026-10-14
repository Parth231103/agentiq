import math
import ast
from typing import List
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from vector_store import similarity_search
from document_loader import load_url


@tool
def search_knowledge_base(query: str) -> str:
    """Searches indexed company files, policies, and documents for relevant information."""
    results = similarity_search(query, top_k=3)
    if not results:
        return "No relevant records found in the knowledge base."

    formatted = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source} | Relevance: {score:.2f}]\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


@tool
def web_search(query: str) -> str:
    """Performs a live web search using DuckDuckGo to find real-time public information, news, or technical references."""
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"No web search results found for query: '{query}'"

        formatted = []
        for i, item in enumerate(results, 1):
            title = item.get("title", "")
            body = item.get("body", "")
            href = item.get("href", "")
            formatted.append(f"[{i}] {title}\nURL: {href}\nSnippet: {body}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
def fetch_web_page(url: str) -> str:
    """Fetches and extracts clean text content from a specified webpage URL."""
    try:
        docs = load_url(url)
        if not docs:
            return f"Failed to retrieve content from {url}"
        content = docs[0].page_content.strip()
        return content[:3000] if len(content) > 3000 else content
    except Exception as e:
        return f"Error loading webpage: {str(e)}"


SAFE_MATH_GLOBALS = {
    "__builtins__": {},
    "math": math,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(expr: str):
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_MATH_GLOBALS:
                    raise ValueError(f"Function call '{node.func.id}' is not allowed.")
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("Access to private attributes is not allowed.")
    return eval(compile(tree, "<string>", "eval"), SAFE_MATH_GLOBALS, {})


@tool
def calculate(expression: str) -> str:
    """Safely evaluates mathematical and numerical expressions like '1500 * 0.15' or 'sqrt(144) + 10'."""
    try:
        cleaned_expr = expression.strip().replace("^", "**")
        result = _safe_eval(cleaned_expr)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


researcher_tools = [search_knowledge_base]
web_tools = [web_search, fetch_web_page]
analyst_tools = [calculate, search_knowledge_base]
all_tools = [search_knowledge_base, web_search, fetch_web_page, calculate]


if __name__ == "__main__":
    print("--- Testing Tool: search_knowledge_base ---")
    kb_res = search_knowledge_base.invoke({"query": "VPN policy"})
    print(kb_res[:200] + "...\n")

    print("--- Testing Tool: web_search ---")
    web_res = web_search.invoke({"query": "LangGraph multi-agent architecture"})
    print(web_res[:250] + "...\n")

    print("--- Testing Tool: calculate ---")
    calc_res1 = calculate.invoke({"expression": "1250 * 0.18 + 500"})
    calc_res2 = calculate.invoke({"expression": "sqrt(256) * 10"})
    print(f"1250 * 0.18 + 500 => {calc_res1}")
    print(f"sqrt(256) * 10 => {calc_res2}")
