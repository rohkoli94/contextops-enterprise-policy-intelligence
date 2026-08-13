import requests
import streamlit as st

from app.config.settings import settings

st.set_page_config(
    page_title="ContextOps",
    page_icon="📄",
)

st.title("ContextOps")
st.caption("Enterprise Policy Intelligence Platform")


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
            st.text(f"Response content: {response.text}")  # show server response
        except requests.RequestException as req_err:
            st.error(f"Request error: {req_err}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")