import re
import time
import uuid
from pathlib import Path

import requests
import streamlit as st

from app.config.settings import settings


st.set_page_config(
    page_title="ContextOps",
    page_icon=str(Path(__file__).with_name("contextops_favicon.png")),
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_STATE = {
    "conversation_id": None,
    "messages": [],
    "selected_source": None,
    "selected_document_id": None,
    "selected_document_name": None,
    "golden_trace_ids": "",
    "user_has_uploaded_document": False,
    "uploaded_document_id": None,
    "uploaded_document_name": None,
    "ingestion_ready": False,
    "pending_question": None,
}

for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state["conversation_id"] is None:
    st.session_state["conversation_id"] = str(uuid.uuid4())


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            /* Metallic Gold + Graphite — modern enterprise luxury */
            --ctx-bg: #FAFAF9;
            --ctx-surface: #FAFAF9;
            --ctx-surface-2: #27272A;
            --ctx-border: #E0C35A;
            --ctx-border-strong: #C5A028;
            --ctx-text: #18181B;
            --ctx-muted: #52525B;
            --ctx-brown: #27272A;
            --ctx-brown-light: #52525B;
            --ctx-gold: #C5A028;
            --ctx-gold-light: #E0C35A;
            --ctx-green: #52525B;
            --ctx-green-light: #F5F5F4;
        }

        .stApp { background: var(--ctx-bg); color: var(--ctx-text); }
        [data-testid="stHeader"] { background: rgba(250,250,249,.98); }
        [data-testid="stToolbar"] { opacity: .82; }

        .block-container {
            width: 100% !important;
            max-width: 1180px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-top: 4.25rem !important;
            padding-bottom: 3rem !important;
            box-sizing: border-box !important;
        }

        /* Brand sits safely below Streamlit's header. */
        .ctx-brand {
            display:flex; align-items:center; gap:11px;
            margin: 0 0 24px;
        }
        .ctx-brand-mark {
            width:40px; height:40px; border-radius:11px;
            background: linear-gradient(135deg, #C5A028, #E0C35A);
            display:flex; align-items:center; justify-content:center;
            color:#FAFAF9; font-size:17px; font-weight:800;
            box-shadow:0 5px 14px rgba(24,24,27,.10);
        }
        .ctx-brand-name {
            color:#18181B !important;
            font-size:23px; font-weight:800; line-height:1.05;
            opacity:1 !important; visibility:visible !important;
            display:block !important;
        }
        .ctx-brand-subtitle {
            color:#52525B !important; font-size:10px; margin-top:3px;
            opacity:1 !important; visibility:visible !important;
            display:block !important;
        }
        .ctx-brand > div:last-child {
            min-width:0; opacity:1 !important; visibility:visible !important;
        }

        .ctx-card {
            background:var(--ctx-surface); border:1px solid var(--ctx-border);
            border-radius:16px; padding:20px;
            box-shadow:0 5px 18px rgba(24,24,27,.05); margin-bottom:16px;
        }
        .ctx-card-title { color:#18181B; font-size:18px; font-weight:800; margin-bottom:4px; }
        .ctx-card-subtitle { color:var(--ctx-muted); font-size:12px; line-height:1.5; margin-bottom:14px; }

        [data-testid="stFileUploader"] {
            border:1px dashed #C5A028; border-radius:13px;
            background:#FAFAF9; padding:12px; margin:6px 0 14px;
        }
        [data-testid="stFileUploader"] section { padding:5px 7px; }
        [data-testid="stFileUploaderDropzone"] { background:transparent; border:0; }
        [data-testid="stFileUploaderDropzoneInstructions"] > div { color:var(--ctx-muted); }
        /* Uploader: hide Streamlit's built-in size/help text completely.
           Keep only the PDF label requested by the product UI. */
        [data-testid="stFileUploaderDropzoneInstructions"] {
            font-size:0 !important;
            color:transparent !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] * {
            font-size:0 !important;
            color:transparent !important;
            visibility:hidden !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"]::after {
            content:"PDF";
            display:block;
            margin-top:6px;
            color:var(--ctx-muted) !important;
            font-size:12px !important;
            line-height:1.3;
            visibility:visible !important;
        }
        [data-testid="stFileUploader"] button { border-radius:8px; font-weight:700; }

        /* Unified enterprise controls */
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] button {
            border-radius:9px !important;
            font-weight:700 !important;
            min-height:36px !important;
            background:var(--ctx-brown) !important;
            color:#fffdf9 !important;
            border:1px solid var(--ctx-brown) !important;
            box-shadow:none !important;
        }
        div[data-testid="stFormSubmitButton"] button:hover,
        div[data-testid="stButton"] button:hover {
            background:var(--ctx-gold) !important;
            border-color:var(--ctx-gold) !important;
            color:#fffdf9 !important;
        }
        div[data-testid="stFormSubmitButton"] button[kind="primary"],
        div[data-testid="stButton"] button[kind="primary"] {
            background:var(--ctx-gold) !important;
            border-color:var(--ctx-gold) !important;
        }

        .ctx-home-title { color:#18181B; font-size:22px; font-weight:800; margin:0 0 4px; }
        .ctx-chat-title { color:#18181B; font-size:19px; font-weight:800; margin:4px 0 3px; }
        .ctx-chat-sub { color:var(--ctx-muted); font-size:11px; margin-bottom:12px; }

        /* Chat area: continuous flow on desktop */
        .ctx-chat-shell {
            background:transparent !important;
            border:0 !important;
            padding:4px 0 0 !important;
            min-height:40px;
        }
        [data-testid="stChatMessage"] {
            padding:.25rem 0 !important;
            margin:9px 0 !important;
            max-width:100% !important;
            background:transparent !important;
            border:0 !important;
            box-shadow:none !important;
            outline:none !important;
        }
        /* Keep the user's question as a compact, right-aligned chat bubble. */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            display:flex !important;
            justify-content:flex-end !important;
            width:100% !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div {
            width:auto !important;
            max-width:64% !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            width:auto !important;
            max-width:100% !important;
            min-width:0 !important;
        }
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] { display:none !important; }
        [data-testid="stChatMessageContent"] {
            font-size:13px !important;
            line-height:1.58 !important;
            width:fit-content !important;
            max-width:min(76%, 760px) !important;
            padding:11px 15px !important;
            border-radius:15px !important;
        }
        /* User message: stronger contrast so it can never disappear into the page. */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            margin-left:0 !important;
            background:#F3E7B0 !important;
            border:1px solid #C5A028 !important;
            color:#18181B !important;
            border-radius:14px !important;
            border-bottom-right-radius:5px !important;
            padding:10px 15px !important;
            min-height:42px !important;
            display:flex !important;
            align-items:center !important;
            box-sizing:border-box !important;
            box-shadow:0 2px 7px rgba(24,24,27,.045) !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
            margin-right:auto !important;
            background:var(--ctx-surface) !important;
            border:1px solid var(--ctx-border) !important;
            color:#343536 !important;
            border-bottom-left-radius:5px !important;
            box-shadow:0 2px 8px rgba(24,24,27,.04) !important;
            display:block !important;
            width:fit-content !important;
            min-height:0 !important;
            height:auto !important;
            padding:12px 15px 13px !important;
            line-height:1.55 !important;
            box-sizing:border-box !important;
            vertical-align:top !important;
        }
        [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
            margin:0 !important;
            padding:0 !important;
            min-height:0 !important;
            height:auto !important;
        }
        [data-testid="stChatMessageContent"] p {
            margin:0 !important;
            padding:0 !important;
            color:inherit !important;
            line-height:1.55 !important;
            min-height:0 !important;
        }
        [data-testid="stChatMessageContent"] p:last-child {
            margin-bottom:0 !important;
            padding-bottom:0 !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] p {
            margin:0 !important;
            padding:0 !important;
            line-height:1.55 !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {
            line-height:1.45 !important;
            margin:0 !important;
            padding-bottom:10px !important;
            box-sizing:border-box !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p:last-child {
            padding-bottom:10px !important;
        }

        /* Source area is intentionally styled as evidence/citation chips, not large buttons. */
        .ctx-source-label {
            color:#52525B; font-size:9px; font-weight:800;
            text-transform:uppercase; letter-spacing:.08em;
            margin:10px 0 5px;
        }
        .ctx-source-chip-wrap { display:flex; gap:6px; flex-wrap:wrap; margin-top:2px; }
        .ctx-source-chip {
            display:inline-flex; align-items:center; gap:5px;
            padding:4px 9px; border-radius:999px;
            background:#FAFAF9; border:1px solid #E0C35A;
            color:#27272A; font-size:10px; font-weight:700;
        }
        .ctx-source-chip::before { content:'↗'; font-size:9px; color:var(--ctx-gold); }

        /* Make Streamlit source buttons visually behave like compact citation chips. */
        .ctx-source-row [data-testid="stButton"] button {
            min-height:28px !important;
            height:28px !important;
            width:auto !important;
            padding:0 10px !important;
            border-radius:999px !important;
            background:#f3eee6 !important;
            border:1px solid #d9cdbd !important;
            color:#735b43 !important;
            font-size:10px !important;
            font-weight:700 !important;
            box-shadow:none !important;
        }
        .ctx-source-row [data-testid="stButton"] button:hover {
            background:#E0C35A !important;
            border-color:#C5A028 !important;
            color:#18181B !important;
        }

        /* Query composer: normal document flow on desktop. It is deliberately
           NOT Streamlit's st.chat_input because Streamlit positions that
           component as a viewport-level composer, which makes desktop chat
           feel detached from the conversation. */
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) {
            margin-top:16px !important;
            padding:10px 12px !important;
            background:var(--ctx-surface) !important;
            border:1px solid var(--ctx-border) !important;
            border-radius:14px !important;
            box-shadow:0 3px 12px rgba(48,45,40,.035) !important;
        }

        /* Deterministic composer sizing: text field takes remaining width,
           send button never escapes the form on narrow screens. */
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) [data-testid="stHorizontalBlock"] {
            display:grid !important;
            grid-template-columns:minmax(0, 1fr) 44px !important;
            width:100% !important;
            min-width:0 !important;
            max-width:100% !important;
            gap:.45rem !important;
            box-sizing:border-box !important;
        }

        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) [data-testid="stHorizontalBlock"] > div {
            min-width:0 !important;
            max-width:100% !important;
            box-sizing:border-box !important;
        }
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) [data-testid="stTextInput"] input {
            min-height:40px !important;
            border-radius:10px !important;
            border:1px solid #E0C35A !important;
            background:#FAFAF9 !important;
            color:#18181B !important;
            caret-color:#C5A028 !important;
            box-shadow:none !important;
        }
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) input::placeholder {
            color:#52525B !important; opacity:1 !important;
        }
        /* Streamlit/browser focus ring: keep the composer clean, without the
           red outline that appears when the input receives focus. */
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) input,
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) input:focus,
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) input:focus-visible,
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) [data-baseweb="input"],
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) [data-baseweb="input"]:focus-within,
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) [data-testid="stTextInput"] > div:focus-within {
            outline:none !important;
            box-shadow:none !important;
        }
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) [data-testid="stTextInput"] > div {
            border:0 !important;
            box-shadow:none !important;
            outline:none !important;
        }
        [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) [data-testid="stFormSubmitButton"] button {
            min-height:40px !important;
            width:44px !important;
            padding:0 !important;
            background:var(--ctx-gold) !important;
            border:1px solid var(--ctx-gold) !important;
            color:#fff !important;
            border-radius:10px !important;
            font-size:16px !important;
        }


        /* Global Streamlit text-input focus cleanup.
           Applies to Home, Q&A and Admin inputs. Streamlit/BaseWeb can add
           a red focus ring/box-shadow on focus; keep the enterprise gold
           border instead. */
        [data-testid="stTextInput"] input,
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextInput"] input:focus-visible,
        [data-testid="stTextInput"] > div,
        [data-testid="stTextInput"] > div:focus-within,
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stTextArea"] textarea:focus-visible,
        [data-testid="stTextArea"] > div,
        [data-testid="stTextArea"] > div:focus-within,
        [data-baseweb="input"],
        [data-baseweb="input"]:focus-within,
        [data-baseweb="textarea"],
        [data-baseweb="textarea"]:focus-within {
            outline: none !important;
            box-shadow: none !important;
        }

        [data-testid="stTextInput"] > div:focus-within,
        [data-testid="stTextArea"] > div:focus-within,
        [data-baseweb="input"]:focus-within,
        [data-baseweb="textarea"]:focus-within {
            border-color: #C5A028 !important;
            outline: none !important;
            box-shadow: 0 0 0 1px #C5A028 !important;
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextInput"] input:focus-visible,
        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stTextArea"] textarea:focus-visible,
        [data-baseweb="input"] input:focus,
        [data-baseweb="textarea"] textarea:focus,
        input:focus,
        input:focus-visible,
        textarea:focus,
        textarea:focus-visible {
            outline: none !important;
            border-color: transparent !important;
            box-shadow: none !important;
        }

        /* Remove the browser/BaseWeb red focus ring from the surrounding
           control as well. Do not add a replacement focus ring. */
        [data-testid="stTextInput"]:focus-within,
        [data-testid="stTextArea"]:focus-within,
        [data-testid="stTextInput"] > div:focus-within,
        [data-testid="stTextArea"] > div:focus-within,
        [data-baseweb="input"]:focus-within,
        [data-baseweb="textarea"]:focus-within,
        [data-baseweb="base-input"]:focus-within,
        [data-testid="stForm"]:has(input:focus),
        [data-testid="stForm"]:has(textarea:focus) {
            outline: none !important;
            box-shadow: none !important;
        }

        .ctx-new-chat-caption { color:var(--ctx-muted); font-size:10px; }
        .ctx-dialog-note { color:var(--ctx-muted); font-size:11px; margin-bottom:10px; }

        .ctx-status-card { background:#FAFAF9; border:1px solid var(--ctx-border); border-radius:14px; padding:15px; box-shadow:0 4px 14px rgba(24,24,27,.04); margin:10px 0 15px; }
        .ctx-status-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }
        .ctx-status-name { color:#18181B; font-weight:800; font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .ctx-status-pill { border-radius:999px; padding:4px 9px; background:var(--ctx-green-light); color:var(--ctx-green); font-size:9px; font-weight:800; white-space:nowrap; }
        .ctx-status-row { display:flex; align-items:center; gap:9px; padding:7px 0; border-bottom:1px solid #E0E0E0; font-size:11px; }
        .ctx-status-row:last-child { border-bottom:0; }
        .ctx-status-dot { width:17px; height:17px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:9px; font-weight:800; flex:0 0 17px; }
        .ctx-status-done { background:var(--ctx-green-light); color:var(--ctx-green); }
        .ctx-status-active { background:#E0C35A; color:var(--ctx-brown); }
        .ctx-status-wait { background:#F4F4F5; color:#71717A; }
        .ctx-status-label { color:#27272A; font-weight:700; }
        .ctx-status-sub { color:#71717A; margin-left:auto; font-size:9px; }
        .ctx-spin { width:11px; height:11px; border:2px solid #E0C35A; border-top-color:#C5A028; border-radius:50%; animation:ctxspin .8s linear infinite; }
        @keyframes ctxspin { to { transform:rotate(360deg); } }
        .ctx-ready { padding:11px 13px; margin-top:11px; border-radius:10px; background:var(--ctx-green-light); border:1px solid #E0C35A; color:#27272A; font-size:11px; font-weight:800; }

        /* Prevent long flex content from shrinking the Streamlit app to a
           narrow desktop-sized column on mobile browsers. */
        html, body, #root {
            width:100% !important;
            min-width:0 !important;
            max-width:100% !important;
            margin:0 !important;
            padding:0 !important;
            overflow-x:hidden !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stAppViewContainer"] .main,
        [data-testid="stAppViewBlockContainer"],
        section.main,
        section.main > div,
        .stMainBlockContainer,
        .block-container {
            width:calc(100% - 40px) !important;
            min-width:0 !important;
            max-width:980px !important;
            margin-left:auto !important;
            margin-right:auto !important;
            box-sizing:border-box !important;
        }

        .ctx-brand > div:last-child {
            min-width:0 !important;
            max-width:100% !important;
            flex:1 1 auto !important;
        }

        @media (max-width: 700px) {
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
                width:fit-content !important;
                max-width:92% !important;
                padding:11px 13px 12px !important;
                display:block !important;
                min-height:0 !important;
                height:auto !important;
            }

            /* ------------------------------------------------------------
               PHONE LAYOUT
               Keep the entire ContextOps page inside the actual viewport.
               Do NOT use fixed positioning for the Q&A composer: Streamlit
               renders forms inside its own layout tree and fixed positioning
               can detach the composer from the phone viewport.
            ------------------------------------------------------------ */

            html,
            body,
            #root,
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > .main,
            [data-testid="stAppViewContainer"] .main,
            section.main {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                margin: 0 !important;
                padding: 0 !important;
                overflow-x: hidden !important;
                box-sizing: border-box !important;
            }

            [data-testid="stAppViewBlockContainer"],
            .stMainBlockContainer,
            .block-container {
                width: calc(100% - 24px) !important;
                min-width: 0 !important;
                max-width: 680px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                padding: 4.1rem 0 2rem !important;
                box-sizing: border-box !important;
                overflow-x: visible !important;
            }

            /* Every Streamlit layout row must be allowed to shrink. */
            [data-testid="stHorizontalBlock"],
            [data-testid="stVerticalBlock"],
            [data-testid="stLayoutWrapper"],
            [data-testid="stElementContainer"],
            [data-testid="stVerticalBlockBorderWrapper"],
            [data-testid="stForm"] {
                min-width: 0 !important;
                max-width: 100% !important;
                box-sizing: border-box !important;
            }

            [data-testid="stHorizontalBlock"] {
                width: 100% !important;
            }

            /* Brand */
            .ctx-brand {
                width: 100% !important;
                max-width: 100% !important;
                display: flex !important;
                align-items: flex-start !important;
                flex-wrap: nowrap !important;
                gap: 9px !important;
                margin: 0 0 16px !important;
                min-width: 0 !important;
                box-sizing: border-box !important;
                overflow: visible !important;
            }

            .ctx-brand-mark {
                width: 34px !important;
                height: 34px !important;
                border-radius: 9px !important;
                font-size: 13px !important;
                flex: 0 0 34px !important;
            }

            .ctx-brand > div:last-child {
                min-width: 0 !important;
                width: 0 !important;
                flex: 1 1 auto !important;
                overflow: visible !important;
            }

            .ctx-brand-name {
                font-size: 19px !important;
                line-height: 1.1 !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
            }

            .ctx-brand-subtitle {
                font-size: 9px !important;
                line-height: 1.35 !important;
                white-space: normal !important;
                overflow-wrap: anywhere !important;
                max-width: 100% !important;
            }

            /* Home/admin cards stay centered within the phone viewport. */
            .ctx-card,
            [data-testid="stVerticalBlockBorderWrapper"] {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 0 !important;
                margin-left: auto !important;
                margin-right: auto !important;
                box-sizing: border-box !important;
                padding: 14px !important;
                border-radius: 13px !important;
                overflow: visible !important;
            }

            .ctx-card-title {
                font-size: 16px !important;
            }

            .ctx-card-subtitle {
                font-size: 11px !important;
            }

            /* Upload form */
            [data-testid="stFileUploader"],
            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploaderDropzone"],
            [data-testid="stFileUploaderDropzone"] > div,
            [data-testid="stTextInput"],
            [data-testid="stTextArea"],
            [data-testid="stForm"],
            [data-testid="stFormSubmitButton"] {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 0 !important;
                margin-left: auto !important;
                margin-right: auto !important;
                box-sizing: border-box !important;
            }

            [data-testid="stFileUploaderDropzoneInstructions"] {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 0 !important;
                overflow: hidden !important;
                text-align: center !important;
                box-sizing: border-box !important;
            }

            /* Q&A header columns must stack naturally instead of creating
               a narrow/hidden second column on small phones. */
            .ctx-chat-title,
            .ctx-chat-sub {
                max-width: 100% !important;
                overflow-wrap: anywhere !important;
            }

            /* Chat bubbles */
            [data-testid="stChatMessage"] {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 0 !important;
                box-sizing: border-box !important;
            }

            [data-testid="stChatMessageContent"] {
                max-width: 88% !important;
                min-width: 0 !important;
                box-sizing: border-box !important;
                overflow-wrap: anywhere !important;
                word-break: break-word !important;
            }

            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div {
                max-width: 84% !important;
                min-width: 0 !important;
            }

            /* Q&A composer: NORMAL FLOW on mobile.
               This is the important fix for the disappearing/teleporting
               composer seen at 425px wide. */
            [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."]) {
                position: static !important;
                left: auto !important;
                right: auto !important;
                top: auto !important;
                bottom: auto !important;
                transform: none !important;
                float: none !important;
                display: block !important;
                width: 100% !important;
                max-width: 100% !important;
                min-width: 0 !important;
                margin: 16px auto 0 !important;
                padding: 7px !important;
                box-sizing: border-box !important;
                background: #FAFAF9 !important;
                border: 1px solid var(--ctx-border) !important;
                border-radius: 13px !important;
                box-shadow: 0 5px 18px rgba(24,24,27,.08) !important;
                z-index: auto !important;
            }

            [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."])
            [data-testid="stHorizontalBlock"] {
                display: grid !important;
                grid-template-columns: minmax(0, 1fr) 42px !important;
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                gap: .35rem !important;
                box-sizing: border-box !important;
            }

            [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."])
            [data-testid="stHorizontalBlock"] > div {
                min-width: 0 !important;
                max-width: 100% !important;
                box-sizing: border-box !important;
            }

            [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."])
            [data-testid="stTextInput"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
            }

            [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."])
            [data-testid="stTextInput"] input {
                width: 100% !important;
                min-width: 0 !important;
                box-sizing: border-box !important;
            }

            [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."])
            [data-testid="stFormSubmitButton"] {
                width: auto !important;
                min-width: 42px !important;
                max-width: 42px !important;
            }

            [data-testid="stForm"]:has(input[placeholder="Ask a question about your document..."])
            [data-testid="stFormSubmitButton"] button {
                width: 42px !important;
                min-width: 42px !important;
                max-width: 42px !important;
                height: 40px !important;
                padding: 0 !important;
            }

            /* Evidence chips */
            .ctx-source-row {
                width: 100% !important;
                max-width: 100% !important;
                overflow-x: auto !important;
                box-sizing: border-box !important;
            }

            .ctx-source-row [data-testid="stButton"] button {
                width: auto !important;
                min-width: max-content !important;
            }

            /* Admin tabs can scroll horizontally rather than pushing the
               entire page outside the phone viewport. */
            [data-baseweb="tab-list"] {
                width: 100% !important;
                max-width: 100% !important;
                overflow-x: auto !important;
                scrollbar-width: none !important;
            }

            [data-baseweb="tab-list"]::-webkit-scrollbar {
                display: none !important;
            }

            code,
            pre,
            [data-testid="stCodeBlock"] {
                max-width: 100% !important;
                overflow-x: auto !important;
                box-sizing: border-box !important;
            }
        }

        @media (max-width: 480px) {
            .block-container {
                width: calc(100% - 20px) !important;
                max-width: 460px !important;
                margin-left:auto !important;
                margin-right:auto !important;
                padding-top: 3.9rem !important;
            }

            .ctx-brand {
                gap: 8px !important;
                margin-bottom: 14px !important;
            }

            .ctx-brand-mark {
                width: 32px !important;
                height: 32px !important;
                flex-basis: 32px !important;
            }

            .ctx-brand-name {
                font-size: 18px !important;
            }

            .ctx-brand-subtitle {
                font-size: 8.5px !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                padding: 12px !important;
            }

            [data-testid="stChatMessageContent"] {
                max-width: 91% !important;
                font-size: 12.5px !important;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def show_request_error(exc: Exception, response=None) -> None:
    if isinstance(exc, requests.HTTPError):
        st.error(f"HTTP error occurred: {exc}")
        if response is not None:
            try:
                st.text(response.text)
            except Exception:
                pass
    elif isinstance(exc, requests.RequestException):
        st.error(f"Request error: {exc}")
    else:
        st.error(f"Unexpected error: {exc}")


def parse_trace_ids(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def render_answer(answer: str, citations: list[dict], message_id: str) -> None:
    if not answer:
        return

    # Keep source markers out of the visible answer. Show them as compact
    # clickable chips below the complete answer instead.
    parts = re.split(r"\[SOURCE\s+(\d+)\]", answer)
    clean_parts = []
    source_numbers = []

    for index, part in enumerate(parts):
        if index % 2 == 0:
            if part.strip():
                clean_parts.append(part.strip())
        else:
            try:
                number = int(part)
                if number not in source_numbers:
                    source_numbers.append(number)
            except ValueError:
                pass

    clean_answer = "\n\n".join(clean_parts).strip()
    if clean_answer:
        st.markdown(clean_answer)

    citation_map = {
        citation.get("source"): citation
        for citation in citations
        if citation.get("source") is not None
    }

    valid_sources = [
        number for number in source_numbers if number in citation_map
    ]

    if valid_sources:
        st.markdown(
            '<div class="ctx-source-label">Evidence</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ctx-source-row">', unsafe_allow_html=True)
        columns = st.columns(min(len(valid_sources), 4))
        for index, source_number in enumerate(valid_sources):
            citation = citation_map[source_number]
            with columns[index % len(columns)]:
                if st.button(
                    f"Source {source_number}",
                    key=f"source_{message_id}_{source_number}",
                    type="secondary",
                    use_container_width=False,
                ):
                    show_source_details(citation)
        st.markdown('</div>', unsafe_allow_html=True)


@st.dialog("Source details", width="small")
def show_source_details(citation: dict) -> None:
    st.markdown('<div class="ctx-dialog-note">Retrieved evidence used to ground this answer.</div>', unsafe_allow_html=True)
    code_fields = [
        ("Document", citation.get("document_id")),
        ("Version", citation.get("document_version_id")),
        ("Chunk", citation.get("chunk_id")),
        ("Retrieval score", citation.get("score")),
        ("Reranker score", citation.get("reranker_score")),
        ("Page Nnumber", citation.get("page_numbers")),
        ("Hierarchy path", citation.get("hierarchy_path")),
    ]
    for label, value in code_fields:
        if value is not None and value != "":
            st.caption(label)
            st.code(str(value), language=None)
    text_fields = [
        # ("Pages", citation.get("page_numbers")),
        # ("Content type", citation.get("content_type")),
        # ("Hierarchy", citation.get("hierarchy_path")),
    ]
    for label, value in text_fields:
        if value:
            st.caption(label)
            st.write(value)
    content = citation.get("content")
    if content:
        st.caption("Retrieved content")
        st.markdown(content)


def upload_document(
    uploaded_file,
    document_name: str,
    categories_input: str,
    tags_input: str,
) -> bool:
    if uploaded_file is None:
        st.warning("Please select a PDF document.")
        return False

    if not document_name.strip():
        st.warning("Please enter a document name.")
        return False

    if not categories_input.strip():
        st.warning("Please enter at least one category.")
        return False

    if not tags_input.strip():
        st.warning("Please enter at least one tag.")
        return False

    categories = [x.strip() for x in categories_input.split(",") if x.strip()]
    tags = [x.strip() for x in tags_input.split(",") if x.strip()]

    if not categories:
        st.warning("Please enter at least one valid category.")
        return False

    if not tags:
        st.warning("Please enter at least one valid tag.")
        return False

    form_data = [("document_name", document_name.strip())]
    form_data.extend(("categories", value) for value in categories)
    form_data.extend(("tags", value) for value in tags)

    uploaded_file.seek(0)

    response = None
    try:
        with st.spinner("Uploading and starting ingestion..."):
            response = requests.post(
                f"{settings.api_base_url}/api/v1/documents",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type or "application/pdf",
                    )
                },
                data=form_data,
                timeout=120,
            )

        response.raise_for_status()
        result = response.json()

        document_id = result.get("document_id") or result.get("id")
        if not document_id:
            try:
                docs = fetch_documents()
                matches = [d for d in docs if d.get("document_name") == document_name.strip()]
                if matches:
                    document_id = matches[-1].get("document_id")
            except Exception:
                pass
        st.session_state["uploaded_document_id"] = document_id
        st.session_state["uploaded_document_name"] = document_name.strip()
        st.session_state["ingestion_ready"] = False
        st.session_state["user_has_uploaded_document"] = True
        st.session_state["messages"] = []
        st.session_state["conversation_id"] = str(uuid.uuid4())
        return True
    except Exception as exc:
        show_request_error(exc, response)
        return False


def fetch_documents() -> list[dict]:
    response = requests.get(
        f"{settings.api_base_url}/api/v1/documents",
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("documents", [])


def ask_question(question: str) -> None:
    normalized_question = question.strip()

    if not normalized_question:
        st.warning("Please enter a question.")
        return

    st.session_state["messages"].append(
        {"role": "user", "content": normalized_question}
    )

    payload = {
        "query": normalized_question,
        "tenant_id": settings.default_tenant_id,
        "conversation_id": st.session_state["conversation_id"],
    }

    response = None

    try:
        with st.spinner("Searching the knowledge base..."):
            response = requests.post(
                f"{settings.api_base_url}/api/v1/query",
                json=payload,
                timeout=120,
            )

        response.raise_for_status()
        result = response.json()

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result.get("answer", "No answer returned."),
                "citations": result.get("citations", []),
                "metadata": result.get("metadata", {}),
            }
        )

        backend_conversation_id = result.get("conversation_id")
        if backend_conversation_id:
            st.session_state["conversation_id"] = backend_conversation_id

        st.rerun()

    except Exception as exc:
        show_request_error(exc, response)


def render_chat() -> None:
    header_left, header_right = st.columns([7, 1.4], vertical_alignment="center")

    with header_left:
        st.markdown(
            '<div class="ctx-chat-title">Ask your document</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ctx-chat-sub">Grounded answers from your uploaded policy.</div>',
            unsafe_allow_html=True,
        )

    with header_right:
        if st.session_state["messages"]:
            if st.button(
                "New chat",
                key="new_chat_top",
                use_container_width=True,
            ):
                st.session_state["conversation_id"] = str(uuid.uuid4())
                st.session_state["messages"] = []
                st.session_state["selected_source"] = None
                st.rerun()

    st.markdown('<div class="ctx-chat-shell">', unsafe_allow_html=True)

    for index, message in enumerate(st.session_state["messages"]):
        role = message.get("role")
        content = message.get("content", "")

        with st.chat_message(role):
            if role == "assistant":
                render_answer(
                    content,
                    message.get("citations", []),
                    f"message_{index}",
                )
            else:
                st.markdown(content)

    st.markdown("</div>", unsafe_allow_html=True)

    # Keep the composer inside the conversation flow on both desktop and mobile.
    # This avoids Streamlit fixed-position layout issues on narrow mobile viewports.
    with st.form("query_composer", clear_on_submit=True, border=False):
        composer_left, composer_send = st.columns([12, 0.8], vertical_alignment="center")
        with composer_left:
            question = st.text_input(
                "Question",
                placeholder="Ask a question about your document...",
                label_visibility="collapsed",
            )
        with composer_send:
            submitted = st.form_submit_button("↑", type="primary")

    if submitted and question:
        # Persist the user message first, then rerun. This makes the
        # submitted question visible immediately, before the backend call.
        normalized_question = question.strip()
        if normalized_question:
            st.session_state["messages"].append(
                {"role": "user", "content": normalized_question}
            )
            st.session_state["pending_question"] = normalized_question
            st.rerun()

    # The next run renders the persisted user bubble before making the API call.
    pending_question = st.session_state.get("pending_question")
    if pending_question:
        st.session_state["pending_question"] = None

        payload = {
            "query": pending_question,
            "tenant_id": settings.default_tenant_id,
            "conversation_id": st.session_state["conversation_id"],
        }

        response = None
        try:
            with st.spinner("Searching the knowledge base..."):
                response = requests.post(
                    f"{settings.api_base_url}/api/v1/query",
                    json=payload,
                    timeout=120,
                )

            response.raise_for_status()
            result = response.json()

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": result.get("answer", "No answer returned."),
                    "citations": result.get("citations", []),
                    "metadata": result.get("metadata", {}),
                }
            )

            backend_conversation_id = result.get("conversation_id")
            if backend_conversation_id:
                st.session_state["conversation_id"] = backend_conversation_id

            st.rerun()

        except Exception as exc:
            show_request_error(exc, response)


def get_document_status(document_id: str | None) -> str:
    if not document_id:
        return "PROCESSING"
    response = requests.get(f"{settings.api_base_url}/api/v1/documents", timeout=15)
    response.raise_for_status()
    for document in response.json().get("documents", []):
        if str(document.get("document_id")) == str(document_id):
            return str(document.get("status") or "PROCESSING").upper()
    return "PROCESSING"


DOCUMENT_STATUS_POLL_SECONDS = 2


def render_ingestion_status(document_id: str | None, document_name: str) -> bool:
    status_placeholder = st.empty()
    started = time.time()
    timeout_seconds = 300
    while time.time() - started < timeout_seconds:
        try:
            status = get_document_status(document_id)
        except Exception:
            status_placeholder.warning("Checking document processing status…")
            time.sleep(DOCUMENT_STATUS_POLL_SECONDS)
            continue
        if status in {"FAILED", "ERROR"}:
            status_placeholder.error("Document processing failed. Please try again.")
            return False
        ready = status in {"INDEXED", "ACTIVE", "COMPLETED", "READY"}
        if ready:
            active_stage = 6
        elif status in {"PROCESSING", "QUEUED"}:
            active_stage = min(5, max(1, int((time.time() - started) // 3) + 1))
        else:
            active_stage = 1
        stages = ["Multimodal extraction completed", "Chunking completed", "Vector DB updated", "Document version activated", "Document processing completed", "Document ready"]
        rows=[]
        for idx,label in enumerate(stages,1):
            if idx < active_stage or active_stage == 6:
                dot='<span class="ctx-status-dot ctx-status-done">✓</span>'; sub='Done'
            elif idx == active_stage:
                dot='<span class="ctx-status-dot ctx-status-active"><span class="ctx-spin"></span></span>'; sub='Processing'
            else:
                dot='<span class="ctx-status-dot ctx-status-wait">•</span>'; sub='Waiting'
            rows.append(f'<div class="ctx-status-row">{dot}<span class="ctx-status-label">{label}</span><span class="ctx-status-sub">{sub}</span></div>')
        ready_html='<div class="ctx-ready">✓ Your document is ready — you can now ask questions.</div>' if ready else ''
        pill="Ready" if ready else "Processing"
        html='<div class="ctx-status-card"><div class="ctx-status-head"><div class="ctx-status-name">'+document_name+'</div><div class="ctx-status-pill">'+pill+'</div></div>'+''.join(rows)+ready_html+'</div>'
        status_placeholder.markdown(html, unsafe_allow_html=True)
        if ready:
            return True
        time.sleep(DOCUMENT_STATUS_POLL_SECONDS)
    status_placeholder.warning("Processing is taking longer than expected. Refresh to check again.")
    return False


def generate_golden_dataset(trace_ids: list[str]) -> None:
    if not trace_ids:
        st.warning("Please enter at least one Langfuse trace ID.")
        return

    response = None

    try:
        with st.spinner("Generating golden dataset and source candidates..."):
            response = requests.post(
                f"{settings.api_base_url}/api/v1/evaluation/golden-dataset/generate-files",
                json={"trace_ids": trace_ids},
                timeout=300,
            )

        response.raise_for_status()
        result = response.json()

        example_count = result.get("example_count", 0)
        candidate_count = result.get("candidate_count", 0)

        st.success(
            f"Golden dataset generated successfully. "
            f"{example_count} examples processed."
        )

        st.write(f"**Golden examples:** {example_count}")
        st.write(f"**Source candidate sets:** {candidate_count}")

        if result.get("golden_dataset_file"):
            st.write("**Golden dataset file**")
            st.code(result["golden_dataset_file"])

        if result.get("golden_source_candidates_file"):
            st.write("**Source candidates file**")
            st.code(result["golden_source_candidates_file"])

        with st.expander("View generated golden dataset"):
            st.json(result.get("golden_dataset", []))

        with st.expander("View generated source candidates"):
            st.json(result.get("source_candidates", []))

    except Exception as exc:
        show_request_error(exc, response)


def render_upload_card(form_key: str, file_key: str | None = None) -> bool:
    with st.form(form_key):
        uploaded_file = st.file_uploader(
            "PDF document",
            type=["pdf"],
            key=file_key,
            label_visibility="collapsed",
        )

        document_name = st.text_input(
            "Document name",
            placeholder="e.g. Personal Loan Policy",
        )

        categories = st.text_input(
            "Categories *",
            placeholder="e.g. Loans, Personal Loan",
        )

        tags = st.text_input(
            "Tags *",
            placeholder="e.g. interest-rate, eligibility",
        )

        upload_clicked = st.form_submit_button(
            "Upload Document",
            use_container_width=True,
            type="primary",
        )

    if not upload_clicked:
        return False

    return upload_document(
        uploaded_file,
        document_name,
        categories,
        tags,
    )


def home_page() -> None:
    inject_styles()
    st.markdown('<div class="ctx-brand"><div class="ctx-brand-mark">C</div><div><div class="ctx-brand-name">ContextOps</div><div class="ctx-brand-subtitle">Policy intelligence · Multimodal extraction · Text · Diagrams · Charts · Images</div></div></div>', unsafe_allow_html=True)
    if not st.session_state["user_has_uploaded_document"]:
        with st.container(border=True):
            st.markdown('<div class="ctx-card-title">Upload PDF</div>', unsafe_allow_html=True)
            st.markdown('<div class="ctx-card-subtitle">Add a document to start asking questions.</div>', unsafe_allow_html=True)
            uploaded=render_upload_card("home_upload_form")
            if uploaded:
                st.rerun()
        return
    document_id=st.session_state.get("uploaded_document_id")
    document_name=st.session_state.get("uploaded_document_name") or "Document"
    if not st.session_state.get("ingestion_ready"):
        if render_ingestion_status(document_id, document_name):
            st.session_state["ingestion_ready"]=True
            st.rerun()
        return
    render_chat()


def admin_page() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="ctx-brand">
            <div class="ctx-brand-mark">C</div>
            <div>
                <div class="ctx-brand-name">ContextOps</div>
                <div class="ctx-brand-subtitle">Enterprise Policy Intelligence</div>
            </div>
        </div>

        <div class="ctx-admin-bar">
            <h1>Administration</h1>
            <p>
                Manage documents, versions, evaluation datasets and active
                knowledge-base documents.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_tab, update_tab, evaluation_tab, documents_tab = st.tabs(
        [
            "Upload Document",
            "Update Version",
            "Golden Dataset",
            "Active Documents",
        ]
    )

    with upload_tab:
        st.markdown(
            '<div class="ctx-card-title">Upload a new document</div>'
            '<div class="ctx-card-subtitle">'
            "Create a new document and its first version."
            "</div>",
            unsafe_allow_html=True,
        )

        uploaded = render_upload_card(
            "admin_upload_document_form",
            file_key="admin_upload_file",
        )

        if uploaded:
            st.rerun()

    with update_tab:
        try:
            documents = fetch_documents()
        except Exception as exc:
            show_request_error(exc)
            documents = []

        if not documents:
            st.info("No active documents available.")
        else:
            options = {
                f"{doc.get('document_name', 'Unnamed')} · "
                f"v{doc.get('current_version', 'N/A')}": doc
                for doc in documents
            }

            selected_label = st.selectbox(
                "Document",
                list(options.keys()),
            )
            selected = options[selected_label]

            st.caption(
                f"Document ID: {selected.get('document_id', 'N/A')}"
            )

            with st.form("admin_update_version_form"):
                version_file = st.file_uploader(
                    "New PDF version",
                    type=["pdf"],
                    key=f"admin_version_{selected['document_id']}",
                )

                update_clicked = st.form_submit_button(
                    "Upload New Version",
                    use_container_width=True,
                    type="primary",
                )

            if update_clicked:
                if version_file is None:
                    st.warning("Please select the updated PDF document.")
                else:
                    version_file.seek(0)
                    response = None

                    try:
                        with st.spinner("Uploading new version..."):
                            response = requests.post(
                                f"{settings.api_base_url}"
                                f"/api/v1/documents/{selected['document_id']}/versions",
                                files={
                                    "file": (
                                        version_file.name,
                                        version_file,
                                        version_file.type or "application/pdf",
                                    )
                                },
                                timeout=120,
                            )

                        response.raise_for_status()
                        result = response.json()

                        st.success(
                            f"Version {result.get('version', 'new')} "
                            "uploaded successfully."
                        )
                        st.rerun()

                    except Exception as exc:
                        show_request_error(exc, response)

    with evaluation_tab:
        st.markdown(
            '<div class="ctx-card-title">Generate golden dataset</div>'
            '<div class="ctx-card-subtitle">'
            "Provide Langfuse trace IDs, one per line."
            "</div>",
            unsafe_allow_html=True,
        )

        st.text_area(
            "Langfuse Trace IDs",
            key="golden_trace_ids",
            placeholder="One trace ID per line",
            height=180,
        )

        trace_ids = parse_trace_ids(
            st.session_state["golden_trace_ids"]
        )

        st.caption(f"{len(trace_ids)} trace ID(s) configured.")

        if st.button(
            "Generate Golden Dataset",
            use_container_width=True,
            type="primary",
        ):
            generate_golden_dataset(trace_ids)

    with documents_tab:
        try:
            documents = fetch_documents()
        except Exception as exc:
            show_request_error(exc)
            documents = []

        if not documents:
            st.info("No active documents uploaded.")
        else:
            st.write(f"**{len(documents)} active document(s)**")

            for document in documents:
                name = document.get("document_name", "Unnamed Document")
                version = document.get("current_version", "N/A")
                categories = document.get("categories", [])
                tags = document.get("tags", [])

                with st.container(border=True):
                    st.markdown(
                        f"**{name}**  \n"
                        f"Version: **v{version}**"
                    )

                    if categories:
                        st.caption(
                            "Categories: " + ", ".join(categories)
                        )

                    if tags:
                        st.caption(
                            "Tags: " + ", ".join(tags)
                        )


pages = [
    st.Page(home_page, title="Home", url_path="", default=True),
    st.Page(admin_page, title="Admin", url_path="admin"),
]

navigation = st.navigation(pages, position="hidden")
navigation.run()
