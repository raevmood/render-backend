# components/graph.py
from langgraph.graph import StateGraph, START, END
from .state import AssistantState
from .retriever import node_retriever
from .conversation import node_conversation
from .generator import node_generator
from .analyzer import node_analyzer

def build_assistant_graph():
    workflow = StateGraph(AssistantState)

    workflow.add_node("EducationalRetriever", node_retriever)
    workflow.add_node("AdaptiveConversationChain", node_conversation)
    workflow.add_node("ContentGenerator", node_generator)
    workflow.add_node("LearningAnalyzer", node_analyzer)

    # seq: retriever -> conversation -> generator -> analyzer
    workflow.add_edge(START, "EducationalRetriever")
    workflow.add_edge("EducationalRetriever", "AdaptiveConversationChain")
    workflow.add_edge("AdaptiveConversationChain", "ContentGenerator")
    workflow.add_edge("ContentGenerator", "LearningAnalyzer")
    workflow.add_edge("LearningAnalyzer", END)

    return workflow.compile()
