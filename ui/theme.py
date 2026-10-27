"""AURORA Design System - Visual Identity, Theme Styling, and Plotly Configurations.

Provides rich gradient accents, glassmorphic card layouts, typography hierarchy,
and cohesive aesthetic styling for the AURORA Multimodal Classroom Intelligence Agent.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# AURORA Design System CSS
AURORA_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Root Variables */
    :root {
        --aurora-bg: #090d16;
        --aurora-card-bg: rgba(15, 23, 42, 0.75);
        --aurora-card-border: rgba(255, 255, 255, 0.08);
        --aurora-cyan: #06b6d4;
        --aurora-teal: #10b981;
        --aurora-indigo: #6366f1;
        --aurora-violet: #8b5cf6;
        --aurora-amber: #f59e0b;
        --aurora-rose: #f43f5e;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    /* Base App Background & Font */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    .stApp {
        background-color: #090d16 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(6, 182, 212, 0.14) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.14) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.10) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        color: #f8fafc !important;
    }

    /* PERMANENTLY HIDE STREAMLIT AUTO-GENERATED SIDEBAR PAGES NAV */
    [data-testid="stSidebarNav"],
    section[data-testid="stSidebar"] nav {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        overflow: hidden !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.45) !important;
    }
    section[data-testid="stSidebar"] div.block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Brand Header */
    .aurora-brand-container {
        padding: 14px 16px;
        margin-bottom: 18px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(139, 92, 246, 0.14) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 4px 20px rgba(6, 182, 212, 0.12);
    }
    .aurora-brand-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .aurora-brand-subtitle {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* CUSTOM SIDEBAR MENU - Transform Radio into Modern Luxury SaaS Navigation */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
        display: none !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
        padding: 0 !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        display: flex !important;
        align-items: center !important;
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 9px !important;
        padding: 10px 14px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.90rem !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(30, 41, 59, 0.85) !important;
        color: #f8fafc !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
        transform: translateX(3px) !important;
    }
    /* Hide the ugly radio dot */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    /* When active / selected */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(6, 182, 212, 0.22) 0%, rgba(99, 102, 241, 0.16) 100%) !important;
        border: 1px solid rgba(6, 182, 212, 0.45) !important;
        border-left: 4px solid #06b6d4 !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.15) !important;
        transform: translateX(2px) !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-top: 2px solid #38bdf8 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-top: 2px solid #818cf8 !important;
        box-shadow: 0 8px 24px rgba(6, 182, 212, 0.15) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #4f46e5 100%) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(51, 65, 85, 0.9) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }

    /* Inputs, Selectboxes, Datepickers */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input, textarea {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: rgba(56, 189, 248, 0.5) !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        background: rgba(6, 182, 212, 0.12) !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Dataframe & Tables */
    div[data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        margin-bottom: 12px !important;
    }

    /* CENTER INTERACTIVE NAVIGATION COMMAND CARDS */
    div.nav-hub {
        background: rgba(11, 17, 32, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 16px 20px 14px 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        margin-bottom: 22px;
    }
    div.nav-hub div[data-testid="stHorizontalBlock"] {
        gap: 12px !important;
        margin-bottom: 10px !important;
    }
    div.nav-hub div[data-testid="stHorizontalBlock"] button {
        min-height: 82px !important;
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 12px !important;
        padding: 12px 10px !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        white-space: pre-line !important;
        line-height: 1.35 !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div.nav-hub div[data-testid="stHorizontalBlock"] button:hover {
        background: rgba(30, 41, 59, 0.95) !important;
        border-color: rgba(56, 189, 248, 0.6) !important;
        color: #38bdf8 !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 28px rgba(6, 182, 212, 0.3) !important;
    }
    div.nav-hub div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.25) 0%, rgba(99, 102, 241, 0.28) 100%) !important;
        border: 2px solid #06b6d4 !important;
        border-top: 3.5px solid #38bdf8 !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 30px rgba(6, 182, 212, 0.45) !important;
        transform: translateY(-2px) !important;
    }

    /* Glowing Pill Badges */
    .badge-present {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
    }
    .badge-late {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fbbf24;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
    }
    .badge-absent {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
    }
    .badge-review {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #818cf8;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
    }
    .badge-offline {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        letter-spacing: 0.04em;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #34d399;
        box-shadow: 0 0 8px #34d399;
        animation: pulseAnimation 2s infinite ease-in-out;
    }
    @keyframes pulseAnimation {
        0% { transform: scale(0.9); opacity: 0.7; }
        50% { transform: scale(1.3); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.7; }
    }
</style>
"""


def apply_aurora_theme() -> None:
    """Inject AURORA CSS stylesheet into the active Streamlit app."""
    st.markdown(AURORA_CSS, unsafe_allow_html=True)


def get_aurora_plotly_layout(height: int = 320, title: str = "") -> Dict[str, Any]:
    """Return a consistent, dark luxury Plotly layout matching the Aurora design system."""
    return dict(
        title=dict(
            text=title,
            font=dict(color="#f8fafc", size=13, family="'Plus Jakarta Sans', sans-serif"),
            x=0.01,
            y=0.98
        ) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.45)",
        font=dict(color="#94a3b8", family="'Plus Jakarta Sans', sans-serif", size=11),
        margin=dict(t=35 if title else 15, b=25, l=35, r=20),
        height=height,
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94a3b8")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(color="#cbd5e1", size=10),
            bgcolor="rgba(0,0,0,0)"
        )
    )


def render_aurora_hero(
    title: str = "AURORA",
    subtitle: str = "Multimodal Classroom Attendance & Engagement Agent",
    badge: str = "OFFLINE AI PIPELINE ACTIVE"
) -> None:
    """Render top hero header with aurora gradients and live status indicator."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 14px; padding: 20px 26px; margin-bottom: 20px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35); position: relative; overflow: hidden;">
            <div style="position: absolute; top: -50px; right: -50px; width: 160px; height: 160px;
                        background: radial-gradient(circle, rgba(6, 182, 212, 0.25) 0%, transparent 70%);"></div>
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.75rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
                                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{title}</span>
                        <span style="font-size: 0.72rem; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3);
                                     color: #38bdf8; padding: 2px 8px; border-radius: 6px; font-weight: 700; letter-spacing: 0.05em;">v2.0 PRO</span>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.94rem; font-weight: 500; margin-top: 4px;">{subtitle}</div>
                </div>
                <div class="badge-offline">
                    <span class="pulse-dot"></span>
                    <span>{badge}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_empty_state(
    title: str = "No Classroom Sessions Recorded Yet",
    message: str = "Get started immediately by generating our synthetic classroom benchmark video or uploading a lecture recording.",
    on_demo_click=None
) -> None:
    """Render a helpful, beautiful first-time onboarding experience."""
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.65); border: 1px dashed rgba(56, 189, 248, 0.3); border-radius: 16px; padding: 36px 30px; text-align: center; margin: 24px 0;">
            <div style="font-size: 2.8rem; margin-bottom: 12px;">🌌</div>
            <h3 style="color: #f8fafc; font-size: 1.4rem; font-weight: 700; margin-bottom: 8px;">{title}</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; max-width: 600px; margin: 0 auto 24px auto;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
