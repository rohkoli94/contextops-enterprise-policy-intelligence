import requests
import streamlit as st

from app.config.settings import settings


st.set_page_config(
    page_title="ContextOps",
    page_icon="📄",
    layout="wide",
)

st.title("ContextOps")
st.caption("Enterprise Policy Intelligence Platform")


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
    st.subheader("💬 Ask a Question")
    st.caption(
        "Ask questions about your enterprise policy documents."
    )

    with st.form("query_form"):
        question = st.text_area(
            "Your Question",
            placeholder=(
                "Example: What is the company's "
                "remote work policy?"
            ),
            height=220,
        )

        ask_clicked = st.form_submit_button(
            "Ask Question",
            use_container_width=True,
        )

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner("Getting answer..."):
                    response = requests.post(
                        f"{settings.api_base_url}/api/v1/query",
                        json={"query": question},
                        timeout=30,
                    )

                response.raise_for_status()
                result = response.json()

                st.divider()
                st.subheader("Answer")

                with st.container(border=True):
                    st.write(result["answer"])

            except requests.HTTPError as http_err:
                st.error(
                    f"HTTP error occurred: {http_err}"
                )
                st.text(
                    f"Response content: {response.text}"
                )

            except requests.RequestException as req_err:
                st.error(f"Request error: {req_err}")

            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ========================================
# RIGHT - DOCUMENT SIDEBAR
# ========================================

with document_column:
    st.subheader("📄 Documents")

    # ------------------------------------
    # Fresh Upload
    # ------------------------------------

    with st.expander(
        "➕ Upload New Document",
        expanded=False,
    ):
        with st.form("upload_document_form"):
            uploaded_file = st.file_uploader(
                "Select PDF",
                type=["pdf"],
            )

            document_name = st.text_input(
                "Document Name",
            )

            categories_input = st.text_input(
                "Categories",
                placeholder="HR, Policies",
            )

            tags_input = st.text_input(
                "Tags",
                placeholder="remote-work, employee",
            )

            upload_document_clicked = (
                st.form_submit_button(
                    "Upload Document",
                    use_container_width=True,
                )
            )

        if upload_document_clicked:
            if uploaded_file is None:
                st.warning("Please select a document.")

            elif not document_name.strip():
                st.warning("Please enter a document name.")

            else:
                try:
                    categories = [
                        category.strip()
                        for category in categories_input.split(",")
                        if category.strip()
                    ]

                    tags = [
                        tag.strip()
                        for tag in tags_input.split(",")
                        if tag.strip()
                    ]

                    form_data = [
                        ("document_name", document_name),
                    ]

                    for category in categories:
                        form_data.append(
                            ("categories", category)
                        )

                    for tag in tags:
                        form_data.append(
                            ("tags", tag)
                        )

                    uploaded_file.seek(0)

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            uploaded_file.type
                            or "application/pdf",
                        )
                    }

                    with st.spinner("Uploading..."):
                        response = requests.post(
                            f"{settings.api_base_url}"
                            f"/api/v1/documents",
                            files=files,
                            data=form_data,
                            timeout=120,
                        )

                    response.raise_for_status()

                    result = response.json()

                    st.success(
                        f"Uploaded successfully! "
                        f"Version {result['version']}"
                    )

                    st.rerun()

                except requests.HTTPError as http_err:
                    st.error(
                        f"HTTP error occurred: {http_err}"
                    )
                    st.text(
                        f"Response content: {response.text}"
                    )

                except requests.RequestException as req_err:
                    st.error(f"Request error: {req_err}")

                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # ------------------------------------
    # Active Documents
    # ------------------------------------

    st.divider()
    st.subheader("Active Documents")

    try:
        response = requests.get(
            f"{settings.api_base_url}/api/v1/documents",
            timeout=30,
        )

        response.raise_for_status()

        documents = response.json()["documents"]

        if not documents:
            st.info("No documents uploaded.")

        else:
            for document in documents:
                with st.container(border=True):
                    st.write(
                        f"**{document['document_name']}**"
                    )

                    st.caption(
                        f"Version v{document['current_version']}"
                    )

                    categories = document.get(
                        "categories",
                        [],
                    )

                    tags = document.get(
                        "tags",
                        [],
                    )

                    if categories:
                        st.caption(
                            "📁 "
                            + ", ".join(categories)
                        )

                    if tags:
                        st.caption(
                            "🏷️ "
                            + ", ".join(tags)
                        )

                    if st.button(
                        "Update Version",
                        key=(
                            f"update_"
                            f"{document['document_id']}"
                        ),
                        use_container_width=True,
                    ):
                        st.session_state.selected_document_id = (
                            document["document_id"]
                        )
                        st.session_state.selected_document_name = (
                            document["document_name"]
                        )
                        st.rerun()

    except requests.RequestException as e:
        st.error(
            f"Unable to load documents: {e}"
        )


# ========================================
# UPDATE DOCUMENT VERSION
# Shows below the main workspace
# ========================================

if "selected_document_id" in st.session_state:
    st.divider()

    st.subheader(
        f"🔄 Update: "
        f"{st.session_state.selected_document_name}"
    )

    with st.form("upload_version_form"):
        version_file = st.file_uploader(
            "Select New PDF Version",
            type=["pdf"],
            key="new_version",
        )

        upload_col, cancel_col = st.columns([3, 1])

        with upload_col:
            upload_version_clicked = (
                st.form_submit_button(
                    "Upload New Version",
                    use_container_width=True,
                )
            )

        with cancel_col:
            cancel_clicked = st.form_submit_button(
                "Cancel",
                use_container_width=True,
            )

    if cancel_clicked:
        del st.session_state.selected_document_id
        del st.session_state.selected_document_name
        st.rerun()

    if upload_version_clicked:
        if version_file is None:
            st.warning("Please select a document.")

        else:
            try:
                version_file.seek(0)

                files = {
                    "file": (
                        version_file.name,
                        version_file,
                        version_file.type
                        or "application/pdf",
                    )
                }

                with st.spinner("Uploading new version..."):
                    response = requests.post(
                        f"{settings.api_base_url}"
                        f"/api/v1/documents/"
                        f"{st.session_state.selected_document_id}"
                        f"/versions",
                        files=files,
                        timeout=120,
                    )

                response.raise_for_status()
                result = response.json()

                st.success(
                    f"Version {result['version']} "
                    f"uploaded successfully!"
                )

                del st.session_state.selected_document_id
                del st.session_state.selected_document_name

                st.rerun()

            except requests.HTTPError as http_err:
                st.error(
                    f"HTTP error occurred: {http_err}"
                )
                st.text(
                    f"Response content: {response.text}"
                )

            except requests.RequestException as req_err:
                st.error(f"Request error: {req_err}")

            except Exception as e:
                st.error(f"Unexpected error: {e}")