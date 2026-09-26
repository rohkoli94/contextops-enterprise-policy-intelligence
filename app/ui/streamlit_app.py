import re
import uuid

import requests
import streamlit as st

from app.config.settings import settings


# ========================================
# Page Configuration
# ========================================

st.set_page_config(
    page_title="ContextOps",
    page_icon="📄",
    layout="wide",
)


# ========================================
# Session State
# ========================================

if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "selected_source" not in st.session_state:
    st.session_state["selected_source"] = None

if "selected_document_id" not in st.session_state:
    st.session_state["selected_document_id"] = None

if "selected_document_name" not in st.session_state:
    st.session_state["selected_document_name"] = None

if "golden_trace_ids" not in st.session_state:
    st.session_state["golden_trace_ids"] = ""


# ========================================
# Helper Functions
# ========================================


def render_answer(
    answer: str,
    citations: list[dict],
    message_id: str,
) -> None:
    """
    Render the answer while converting [SOURCE n]
    references into clickable buttons.
    """

    if not answer:
        return

    parts = re.split(
        r"(\[SOURCE\s+\d+\])",
        answer,
    )

    citation_map = {
        citation.get("source"): citation
        for citation in citations
        if citation.get("source") is not None
    }

    for part_index, part in enumerate(parts):

        source_match = re.fullmatch(
            r"\[SOURCE\s+(\d+)\]",
            part.strip(),
        )

        if source_match:

            source_number = int(
                source_match.group(1)
            )

            citation = citation_map.get(
                source_number
            )

            if citation:

                button_key = (
                    f"source_"
                    f"{message_id}_"
                    f"{part_index}_"
                    f"{source_number}"
                )

                if st.button(
                    f"🔗 [SOURCE {source_number}]",
                    key=button_key,
                ):
                    st.session_state[
                        "selected_source"
                    ] = citation

                    st.rerun()

            else:
                st.markdown(part)

        else:

            if part:
                st.markdown(part)


def show_source_details(
    citation: dict,
) -> None:
    """
    Display the current citation fields.
    No PDF opening logic.
    """

    st.subheader("📌 Source Details")

    source_number = citation.get(
        "source"
    )

    st.markdown(
        f"**[SOURCE {source_number}]**"
    )

    st.divider()

    document_id = citation.get(
        "document_id"
    )

    document_version_id = citation.get(
        "document_version_id"
    )

    chunk_id = citation.get(
        "chunk_id"
    )

    score = citation.get(
        "score"
    )

    reranker_score = citation.get(
        "reranker_score"
    )

    st.write(
        "**Document ID**"
    )

    st.code(
        str(document_id)
        if document_id
        else "N/A"
    )

    st.write(
        "**Document Version ID**"
    )

    st.code(
        str(document_version_id)
        if document_version_id
        else "N/A"
    )

    st.write(
        "**Chunk ID**"
    )

    st.code(
        str(chunk_id)
        if chunk_id
        else "N/A"
    )

    st.write(
        "**Retrieval Score**"
    )

    st.code(
        str(score)
        if score is not None
        else "N/A"
    )

    st.write(
        "**Reranker Score**"
    )

    st.code(
        str(reranker_score)
        if reranker_score is not None
        else "N/A"
    )

    page_numbers = citation.get(
        "page_numbers"
    )

    content_type = citation.get(
        "content_type"
    )

    hierarchy_path = citation.get(
        "hierarchy_path"
    )

    content = citation.get(
        "content"
    )

    if page_numbers:

        st.write(
            "**Pages**"
        )

        st.write(
            page_numbers
        )

    if content_type:

        st.write(
            "**Content Type**"
        )

        st.write(
            content_type
        )

    if hierarchy_path:

        st.write(
            "**Hierarchy**"
        )

        st.write(
            hierarchy_path
        )

    if content:

        st.write(
            "**Content**"
        )

        with st.expander(
            "View retrieved content"
        ):
            st.write(
                content
            )

    if st.button(
        "✕ Close Source",
        key="close_source",
        use_container_width=True,
    ):

        st.session_state[
            "selected_source"
        ] = None

        st.rerun()


def generate_golden_dataset(
    trace_ids: list[str],
) -> None:
    """
    Generate both golden dataset files from
    the supplied Langfuse trace IDs.

    Backend endpoint:
    POST /api/v1/evaluation/golden-dataset/generate-files
    """

    if not trace_ids:

        st.warning(
            "Please enter at least one Langfuse trace ID."
        )

        return

    payload = {
        "trace_ids": trace_ids
    }

    try:

        with st.spinner(
            "Generating golden dataset and source candidates..."
        ):

            response = requests.post(
                (
                    f"{settings.api_base_url}"
                    f"/api/v1/evaluation/"
                    f"golden-dataset/generate-files"
                ),
                json=payload,
                timeout=300,
            )

        response.raise_for_status()

        result = response.json()

        example_count = result.get(
            "example_count",
            0,
        )

        candidate_count = result.get(
            "candidate_count",
            0,
        )

        golden_dataset_file = result.get(
            "golden_dataset_file",
            "",
        )

        source_candidates_file = result.get(
            "golden_source_candidates_file",
            "",
        )

        st.success(
            (
                "Golden dataset generated successfully. "
                f"{example_count} examples processed."
            )
        )

        st.write(
            f"**Golden examples:** {example_count}"
        )

        st.write(
            f"**Source candidate sets:** {candidate_count}"
        )

        if golden_dataset_file:

            st.write(
                "**Golden dataset file**"
            )

            st.code(
                golden_dataset_file
            )

        if source_candidates_file:

            st.write(
                "**Source candidates file**"
            )

            st.code(
                source_candidates_file
            )

        with st.expander(
            "View generated golden dataset",
            expanded=False,
        ):

            st.json(
                result.get(
                    "golden_dataset",
                    [],
                )
            )

        with st.expander(
            "View generated source candidates",
            expanded=False,
        ):

            st.json(
                result.get(
                    "source_candidates",
                    [],
                )
            )

    except requests.HTTPError as http_err:

        st.error(
            f"HTTP error occurred: {http_err}"
        )

        try:

            st.text(
                response.text
            )

        except Exception:
            pass

    except requests.RequestException as req_err:

        st.error(
            f"Request error: {req_err}"
        )

    except Exception as exc:

        st.error(
            f"Unexpected error: {exc}"
        )


# ========================================
# Header
# ========================================

st.title("ContextOps")

st.caption(
    "Enterprise Policy Intelligence Platform"
)


# ========================================
# Header Actions
# ========================================

header_col1, header_col2, header_col3 = (
    st.columns(
        [5, 2.5, 2.5]
    )
)

with header_col1:

    st.caption(
        "Multimodal enterprise policy intelligence"
    )


with header_col2:

    if st.button(
        "🧪 Generate Golden Dataset",
        use_container_width=True,
    ):

        trace_id_text = (
            st.session_state.get(
                "golden_trace_ids",
                "",
            )
        )

        trace_ids = [
            trace_id.strip()
            for trace_id in trace_id_text.splitlines()
            if trace_id.strip()
        ]

        generate_golden_dataset(
            trace_ids=trace_ids
        )


with header_col3:

    if st.button(
        "＋ New Chat",
        use_container_width=True,
    ):

        st.session_state[
            "conversation_id"
        ] = str(uuid.uuid4())

        st.session_state[
            "messages"
        ] = []

        st.session_state[
            "selected_source"
        ] = None

        st.rerun()


# ========================================
# Golden Dataset Configuration
# ========================================

with st.expander(
    "🧪 Golden Dataset Configuration",
    expanded=False,
):

    st.caption(
        (
            "Enter one Langfuse trace ID per line. "
            "You can add any number of trace IDs."
        )
    )

    st.text_area(
        "Langfuse Trace IDs",
        key="golden_trace_ids",
        placeholder=(
            "Paste one Langfuse trace ID per line.\n\n"
            "Example:\n"
            "8416eeadc888ead049d6734e4dd6b62d\n"
            "8a24ef6a9c362bb05f23a2034edfa25f\n"
            "10a007d8383f475ffc72cb4153ba1734\n"
            "48c99001c5cfbeea97ffbf2b5b539c5e"
        ),
        height=180,
        help=(
            "Each non-empty line is treated as one "
            "Langfuse trace ID."
        ),
    )

    trace_id_text = (
        st.session_state.get(
            "golden_trace_ids",
            "",
        )
    )

    configured_trace_ids = [
        trace_id.strip()
        for trace_id in trace_id_text.splitlines()
        if trace_id.strip()
    ]

    st.caption(
        f"{len(configured_trace_ids)} trace ID(s) configured."
    )


st.caption(
    f"Conversation ID: "
    f"`{st.session_state['conversation_id']}`"
)


st.divider()


# ========================================
# Main Workspace
# Query: 70% | Documents: 30%
# ========================================

query_column, document_column = st.columns(
    [7, 3],
    gap="large",
)


# ========================================
# LEFT - QUERY WORKSPACE
# ========================================

with query_column:

    st.subheader(
        "💬 Ask a Question"
    )

    st.caption(
        "Ask questions about your enterprise policy documents."
    )


    # ====================================
    # Conversation History
    # ====================================

    if st.session_state["messages"]:

        for message_index, message in enumerate(
            st.session_state["messages"]
        ):

            role = message.get(
                "role"
            )

            content = message.get(
                "content",
                "",
            )

            with st.chat_message(
                role
            ):

                if role == "assistant":

                    render_answer(
                        answer=content,
                        citations=message.get(
                            "citations",
                            [],
                        ),
                        message_id=(
                            f"message_"
                            f"{message_index}"
                        ),
                    )

                else:

                    st.markdown(
                        content
                    )


    # ====================================
    # Question Form
    # ====================================

    with st.form(
        "query_form",
        clear_on_submit=True,
    ):

        question = st.text_area(
            "Your Question",
            placeholder=(
                "Example: What financial "
                "documents must be obtained "
                "for a business loan application?"
            ),
            height=120,
        )

        tenant_id = st.text_input(
            "Tenant ID",
            value=settings.default_tenant_id,
            placeholder="Enter tenant ID",
        )

        ask_clicked = st.form_submit_button(
            "Ask Question",
            use_container_width=True,
        )


    # ====================================
    # Ask Question
    # ====================================

    if ask_clicked:

        normalized_question = (
            question.strip()
        )

        normalized_tenant = (
            tenant_id.strip()
        )

        if not normalized_question:

            st.warning(
                "Please enter a question."
            )

        elif not normalized_tenant:

            st.warning(
                "Please enter a tenant ID."
            )

        else:

            st.session_state[
                "messages"
            ].append(
                {
                    "role": "user",
                    "content": normalized_question,
                }
            )

            payload = {
                "query": normalized_question,
                "tenant_id": normalized_tenant,
                "conversation_id": (
                    st.session_state[
                        "conversation_id"
                    ]
                ),
            }

            try:

                with st.spinner(
                    "Getting answer..."
                ):

                    response = requests.post(
                        (
                            f"{settings.api_base_url}"
                            f"/api/v1/query"
                        ),
                        json=payload,
                        timeout=120,
                    )

                response.raise_for_status()

                result = response.json()

                st.session_state[
                    "messages"
                ].append(
                    {
                        "role": "assistant",
                        "content": result.get(
                            "answer",
                            "No answer returned.",
                        ),
                        "citations": result.get(
                            "citations",
                            [],
                        ),
                        "metadata": result.get(
                            "metadata",
                            {},
                        ),
                    }
                )

                backend_conversation_id = (
                    result.get(
                        "conversation_id"
                    )
                )

                if backend_conversation_id:

                    st.session_state[
                        "conversation_id"
                    ] = (
                        backend_conversation_id
                    )

                st.rerun()

            except requests.HTTPError as http_err:

                st.error(
                    f"HTTP error occurred: "
                    f"{http_err}"
                )

                try:

                    st.text(
                        response.text
                    )

                except Exception:
                    pass

            except requests.RequestException as req_err:

                st.error(
                    f"Request error: "
                    f"{req_err}"
                )

            except Exception as exc:

                st.error(
                    f"Unexpected error: "
                    f"{exc}"
                )


# ========================================
# RIGHT - DOCUMENT SIDEBAR
# ========================================

with document_column:

    st.subheader(
        "📄 Documents"
    )


    # ====================================
    # Selected Source
    # ====================================

    if st.session_state[
        "selected_source"
    ]:

        with st.container(
            border=True
        ):

            show_source_details(
                st.session_state[
                    "selected_source"
                ]
            )


    # ====================================
    # Upload New Document
    # ====================================

    with st.expander(
        "➕ Upload New Document",
        expanded=False,
    ):

        with st.form(
            "upload_document_form"
        ):

            uploaded_file = (
                st.file_uploader(
                    "Select PDF",
                    type=["pdf"],
                )
            )

            document_name = st.text_input(
                "Document Name",
            )

            categories_input = (
                st.text_input(
                    "Categories",
                    placeholder=(
                        "Business Loan, Home Loan"
                    ),
                )
            )

            tags_input = st.text_input(
                "Tags",
                placeholder=(
                    "under-writing, credit"
                ),
            )

            upload_document_clicked = (
                st.form_submit_button(
                    "Upload Document",
                    use_container_width=True,
                )
            )


        if upload_document_clicked:

            if uploaded_file is None:

                st.warning(
                    "Please select a document."
                )

            elif not document_name.strip():

                st.warning(
                    "Please enter a document name."
                )

            else:

                try:

                    categories = [
                        category.strip()
                        for category in (
                            categories_input.split(
                                ","
                            )
                        )
                        if category.strip()
                    ]

                    tags = [
                        tag.strip()
                        for tag in (
                            tags_input.split(
                                ","
                            )
                        )
                        if tag.strip()
                    ]

                    form_data = [
                        (
                            "document_name",
                            document_name.strip(),
                        )
                    ]

                    for category in categories:

                        form_data.append(
                            (
                                "categories",
                                category,
                            )
                        )

                    for tag in tags:

                        form_data.append(
                            (
                                "tags",
                                tag,
                            )
                        )

                    uploaded_file.seek(
                        0
                    )

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            uploaded_file.type
                            or "application/pdf",
                        )
                    }

                    with st.spinner(
                        "Uploading..."
                    ):

                        response = (
                            requests.post(
                                (
                                    f"{settings.api_base_url}"
                                    f"/api/v1/documents"
                                ),
                                files=files,
                                data=form_data,
                                timeout=120,
                            )
                        )

                    response.raise_for_status()

                    result = response.json()

                    st.success(
                        (
                            "Uploaded successfully! "
                            f"Version {result['version']}"
                        )
                    )

                    st.rerun()

                except requests.HTTPError as http_err:

                    st.error(
                        f"HTTP error occurred: "
                        f"{http_err}"
                    )

                    try:

                        st.text(
                            response.text
                        )

                    except Exception:
                        pass

                except requests.RequestException as req_err:

                    st.error(
                        f"Request error: "
                        f"{req_err}"
                    )

                except Exception as exc:

                    st.error(
                        f"Unexpected error: "
                        f"{exc}"
                    )


    # ====================================
    # Active Documents
    # ====================================

    st.divider()

    st.subheader(
        "Active Documents"
    )

    try:

        response = requests.get(
            (
                f"{settings.api_base_url}"
                f"/api/v1/documents"
            ),
            timeout=30,
        )

        response.raise_for_status()

        documents = response.json().get(
            "documents",
            [],
        )

        if not documents:

            st.info(
                "No documents uploaded."
            )

        else:

            for document in documents:

                document_id = document.get(
                    "document_id"
                )

                document_name = document.get(
                    "document_name",
                    "Unnamed Document",
                )

                current_version = document.get(
                    "current_version",
                    "N/A",
                )

                with st.container(
                    border=True
                ):

                    st.write(
                        f"**{document_name}**"
                    )

                    st.caption(
                        (
                            f"Version "
                            f"v{current_version}"
                        )
                    )

                    categories = (
                        document.get(
                            "categories",
                            [],
                        )
                    )

                    tags = document.get(
                        "tags",
                        [],
                    )

                    if categories:

                        st.caption(
                            "📁 "
                            + ", ".join(
                                categories
                            )
                        )

                    if tags:

                        st.caption(
                            "🏷️ "
                            + ", ".join(
                                tags
                            )
                        )

                    if st.button(
                        "Update Version",
                        key=(
                            f"update_"
                            f"{document_id}"
                        ),
                        use_container_width=True,
                    ):

                        st.session_state[
                            "selected_document_id"
                        ] = document_id

                        st.session_state[
                            "selected_document_name"
                        ] = document_name

                        st.rerun()

    except requests.RequestException as exc:

        st.error(
            (
                "Unable to load documents: "
                f"{exc}"
            )
        )


# ========================================
# UPDATE DOCUMENT VERSION
# ========================================

if (
    st.session_state.get(
        "selected_document_id"
    )
):

    st.divider()

    st.subheader(
        "🔄 Update: "
        + str(
            st.session_state.get(
                "selected_document_name",
                "",
            )
        )
    )

    with st.form(
        "upload_version_form"
    ):

        version_file = st.file_uploader(
            "Select New PDF Version",
            type=["pdf"],
            key="new_version",
        )

        upload_col, cancel_col = (
            st.columns([3, 1])
        )

        with upload_col:

            upload_version_clicked = (
                st.form_submit_button(
                    "Upload New Version",
                    use_container_width=True,
                )
            )

        with cancel_col:

            cancel_clicked = (
                st.form_submit_button(
                    "Cancel",
                    use_container_width=True,
                )
            )


    # ====================================
    # Cancel Update
    # ====================================

    if cancel_clicked:

        st.session_state[
            "selected_document_id"
        ] = None

        st.session_state[
            "selected_document_name"
        ] = None

        st.rerun()


    # ====================================
    # Upload Version
    # ====================================

    if upload_version_clicked:

        if version_file is None:

            st.warning(
                "Please select a document."
            )

        else:

            try:

                version_file.seek(
                    0
                )

                files = {
                    "file": (
                        version_file.name,
                        version_file,
                        version_file.type
                        or "application/pdf",
                    )
                }

                with st.spinner(
                    "Uploading new version..."
                ):

                    response = requests.post(
                        (
                            f"{settings.api_base_url}"
                            f"/api/v1/documents/"
                            f"{st.session_state['selected_document_id']}"
                            f"/versions"
                        ),
                        files=files,
                        timeout=120,
                    )

                response.raise_for_status()

                result = response.json()

                st.success(
                    (
                        f"Version {result['version']} "
                        "uploaded successfully!"
                    )
                )

                st.session_state[
                    "selected_document_id"
                ] = None

                st.session_state[
                    "selected_document_name"
                ] = None

                st.rerun()

            except requests.HTTPError as http_err:

                st.error(
                    f"HTTP error occurred: "
                    f"{http_err}"
                )

                try:

                    st.text(
                        response.text
                    )

                except Exception:
                    pass

            except requests.RequestException as req_err:

                st.error(
                    f"Request error: "
                    f"{req_err}"
                )

            except Exception as exc:

                st.error(
                    f"Unexpected error: "
                    f"{exc}"
                )