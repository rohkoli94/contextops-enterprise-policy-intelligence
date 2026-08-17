import requests
import streamlit as st

from app.config.settings import settings


st.set_page_config(
    page_title="ContextOps",
    page_icon="📄",
)

st.title("ContextOps")
st.caption("Enterprise Policy Intelligence Platform")


# ========================================
# Document Upload
# ========================================

st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Select a document",
    type=["pdf"],
)

document_name = st.text_input(
    "Document name",
    placeholder="Example: Remote Work Policy",
)

categories_input = st.text_input(
    "Categories (comma-separated)",
    placeholder="Example: HR, Policies",
)

tags_input = st.text_input(
    "Tags (comma-separated)",
    placeholder="Example: remote-work, employee",
)


if st.button("Upload Document"):
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

            # Reset the file stream before sending.
            uploaded_file.seek(0)

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    uploaded_file.type or "application/pdf",
                )
            }

            response = requests.post(
                f"{settings.api_base_url}/api/v1/documents",
                files=files,
                data=form_data,
                timeout=120,
            )

            response.raise_for_status()

            result = response.json()

            st.success("Document uploaded successfully!")

            st.write("Document ID:", result["document_id"])
            st.write(
                "Document Version ID:",
                result["document_version_id"],
            )
            st.write("Version:", result["version"])
            st.write("Status:", result["status"])

        except requests.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err}")
            st.text(f"Response content: {response.text}")

        except requests.RequestException as req_err:
            st.error(f"Request error: {req_err}")

        except Exception as e:
            st.error(f"Unexpected error: {e}")


# ========================================
# Query
# ========================================

st.divider()

st.subheader("Ask a Question")

question = st.text_area(
    "Ask a question about enterprise policies",
    placeholder="Example: What is the company's remote work policy?",
)


if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            response = requests.post(
                f"{settings.api_base_url}/api/v1/query",
                json={"query": question},
                timeout=30,
            )

            response.raise_for_status()

            result = response.json()

            st.subheader("Answer")
            st.write(result["answer"])

        except requests.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err}")
            st.text(f"Response content: {response.text}")

        except requests.RequestException as req_err:
            st.error(f"Request error: {req_err}")

        except Exception as e:
            st.error(f"Unexpected error: {e}")