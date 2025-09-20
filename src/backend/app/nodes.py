"""Core graph nodes for planning, acting, and observing."""

from .state import AppState


def plan_node(state: AppState) -> AppState:
    """Plan what to do based on the user query and recalled memories."""
    # TODO: Implement planning logic
    state.messages.append({
        "role": "system",
        "content": f"Planning response for: {state.user_query}"
    })
    return state


def act_node(state: AppState) -> AppState:
    """Execute the planned action."""
    # TODO: Implement action logic (e.g., call OpenAI API)
    state.messages.append({
        "role": "assistant",
        "content": f"Acting on query: {state.user_query}"
    })
    return state


def observe_node(state: AppState) -> AppState:
    """Observe the results and prepare final response."""
    # TODO: Implement observation logic
    state.result = f"Completed processing: {state.user_query}"
    return state