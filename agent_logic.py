"""
LangChain tool-calling agent for the Tel Aviv Shelter Locator.

Defines the geocode_address and find_shelters tools, system prompt,
and cached AgentExecutor. API keys are read from environment or Streamlit secrets.
"""

from __future__ import annotations

import os

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from data_engine import (
    find_nearest_shelters_from_coords,
    load_addresses_data,
    load_shelters_data,
    lookup_address_offline,
)


def _get_openai_api_key() -> str | None:
    """Resolve OpenAI API key from environment or Streamlit secrets (never hardcoded)."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


@tool
def geocode_address(street_name: str, house_number: int | str) -> dict | str:
    """
    Look up coordinates for a Hebrew street name and house number in Tel Aviv.
    Use when the user gives their location (e.g. street name and number).
    street_name: Hebrew street name (e.g. דיזנגוף). house_number: number as int or string.
    Returns a dict with lat, lon and found_address, or an error string if not found.
    """
    df_addresses = load_addresses_data()
    lat, lon, display_address, _used_fallback, _fallback_num = lookup_address_offline(
        df_addresses, street_name.strip(), str(house_number).strip()
    )
    if lat is None or lon is None:
        return "Address not found in the municipal database. Please verify the street name and house number."
    return {"lat": lat, "lon": lon, "found_address": display_address or f"{street_name} {house_number}, Tel Aviv"}


@tool
def find_shelters(lat: float, lon: float) -> list:
    """
    Find the 3 closest emergency shelters to the given coordinates in Tel Aviv.
    Returns a list of up to 3 shelters with distance_m, address, lat, lon, size.
    """
    df = load_shelters_data()
    top_3_list, err = find_nearest_shelters_from_coords(lat, lon, df)
    if err:
        return []
    if top_3_list:
        try:
            st.session_state.latest_map_data = {
                "user_lat": lat,
                "user_lon": lon,
                "shelters": top_3_list,
            }
        except Exception:
            pass
    return top_3_list


SYSTEM_PROMPT = (
    "You are a Tel Aviv tactical emergency assistant. The user will state their location and situation. "
    "Reply ONLY in English. Provide ONLY a short, tactical opening sentence in English. "
    "Example: 'Here are the closest shelters. Stay safe.' Do not use Hebrew. "
    "If the user provides exact latitude and longitude coordinates, SKIP the 'geocode_address' tool entirely "
    "and directly use the 'find_shelters' tool with those exact coordinates. "
    "Otherwise: 1. Use 'geocode_address' to find coordinates (street_name in Hebrew, house_number as string). "
    "2. Use 'find_shelters' to find the 3 closest safe locations. "
    "CRITICAL: DO NOT output the list of shelters or their distances yourself. The UI will handle displaying the list. "
    "Do not output raw JSON."
)


@st.cache_resource
def create_agent_executor():
    """
    Build and cache the LangChain tool-calling agent and executor.

    Uses OPENAI_API_KEY from environment or st.secrets. Returns an AgentExecutor
    that runs geocode_address and find_shelters tools.
    """
    api_key = _get_openai_api_key()
    if not api_key:
        return None
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    tools = [geocode_address, find_shelters]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
