import streamlit as st
import os
import base64
import json
import time
import httpx
from typing import Any, cast

# The current app calls the OCR REST endpoint directly via httpx, so a top-level
# SDK client import is not required here. Newer mistralai releases may expose a
# namespace package without `from mistralai import Mistral`, which would make the
# app fail during startup even though the package is installed correctly.

# DocumentURLChunk & ImageURLChunk: try the most specific SDK path first, then
# fall back to older public import locations.
try:
    from mistralai.client.models import DocumentURLChunk, ImageURLChunk  # type: ignore[import]
    _USE_TYPED_CHUNKS = True
except ImportError:
    try:
        from mistralai import DocumentURLChunk, ImageURLChunk  # type: ignore[import]
        _USE_TYPED_CHUNKS = True
    except ImportError:
        try:
            from mistralai.models import DocumentURLChunk, ImageURLChunk  # type: ignore[import]
            _USE_TYPED_CHUNKS = True
        except ImportError:
            _USE_TYPED_CHUNKS = False


def _make_document_url_chunk(url: str) -> Any:
    """Return a DocumentURLChunk if available, otherwise a plain dict."""
    if _USE_TYPED_CHUNKS:
        return DocumentURLChunk(document_url=url)  # type: ignore[call-arg]
    return {"type": "document_url", "document_url": url}


def _make_image_url_chunk(url: str) -> Any:
    """Return an ImageURLChunk if available, otherwise a plain dict."""
    if _USE_TYPED_CHUNKS:
        return ImageURLChunk(image_url=url)  # type: ignore[call-arg]
    return {"type": "image_url", "image_url": url}

def create_download_link(data: str, filetype: str, filename: str) -> None:
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:{filetype};base64,{b64}" download="{filename}">Download {filename}</a>'
    st.markdown(href, unsafe_allow_html=True)

def _chunk_to_dict(document: Any) -> dict:
    """Convert a typed chunk or plain dict to a JSON-serialisable dict."""
    if isinstance(document, dict):
        return document
    if hasattr(document, "document_url"):
        return {"type": "document_url", "document_url": document.document_url}
    if hasattr(document, "image_url"):
        return {"type": "image_url", "image_url": document.image_url}
    # Fallback: try __dict__
    return vars(document)


def call_mistral_ocr(api_key: str, document: Any) -> Any:
    """
    Call the Mistral OCR API via a direct HTTP request.

    This avoids relying on `client.ocr`, which is not present in all SDK
    versions, while keeping identical behaviour and response structure.
    """
    payload = {
        "model": "mistral-ocr-latest",
        "document": _chunk_to_dict(document),
        "include_image_base64": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = httpx.post(
        "https://api.mistral.ai/v1/ocr",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


st.set_page_config(layout="wide", page_title="Mistral OCR App", page_icon="🖥️")
st.title("Mistral OCR App")
st.markdown("<h3 style color: white;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)
with st.expander("Expand Me"):
    st.markdown("""
    This application allows you to extract information from pdf/image based on Mistral OCR. Built by AI Anytime.
    """)

# 1. API Key Input
api_key = st.text_input("Enter your Mistral API Key", type="password")
if not api_key:
    st.info("Please enter your API key to continue.")
    st.stop()

# Initialize session state variables for persistence
if "ocr_result" not in st.session_state:
    st.session_state["ocr_result"] = []
if "preview_src" not in st.session_state:
    st.session_state["preview_src"] = []
if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = []

# 2. Choose file type: PDF or Image
file_type = st.radio("Select file type", ("PDF", "Image"))

# 3. Select source type: URL or Local Upload
source_type = st.radio("Select source type", ("URL", "Local Upload"))

input_url = ""
uploaded_files = []

if source_type == "URL":
    input_url = st.text_area("Enter one or multiple URLs (separate with new lines)")
else:
    uploaded_files = st.file_uploader("Upload one or more files", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

# 4. Process Button & OCR Handling
if st.button("Process"):
    if source_type == "URL" and not input_url.strip():
        st.error("Please enter at least one valid URL.")
    elif source_type == "Local Upload" and not uploaded_files:
        st.error("Please upload at least one file.")
    else:
        st.session_state["ocr_result"] = []
        st.session_state["preview_src"] = []
        st.session_state["image_bytes"] = []

        if source_type == "URL":
            url_sources: list[str] = input_url.split("\n")
            for url in url_sources:
                url = url.strip()
                if not url:
                    continue

                if file_type == "PDF":
                    document: Any = _make_document_url_chunk(url)
                    preview_src = url
                else:
                    document = _make_image_url_chunk(url)
                    preview_src = url

                with st.spinner(f"Processing {url}..."):
                    try:
                        # ✅ FIX: use direct HTTP call instead of client.ocr.process()
                        ocr_response = call_mistral_ocr(api_key, document)
                        time.sleep(1)

                        pages = ocr_response.get("pages", [])
                        result_text = (
                            "\n\n".join(
                                page["markdown"]
                                for page in pages
                                if "markdown" in page
                            )
                            or "No result found."
                        )
                    except Exception as e:
                        result_text = f"Error extracting result: {e}"

                    st.session_state["ocr_result"].append(result_text)
                    st.session_state["preview_src"].append(preview_src)

        else:
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                mime_type: str = uploaded_file.type
                file_name: str = uploaded_file.name

                if file_type == "PDF":
                    encoded_pdf = base64.b64encode(file_bytes).decode("utf-8")
                    document = _make_document_url_chunk(f"data:application/pdf;base64,{encoded_pdf}")
                    preview_src = f"data:application/pdf;base64,{encoded_pdf}"
                else:
                    encoded_image = base64.b64encode(file_bytes).decode("utf-8")
                    document = _make_image_url_chunk(f"data:{mime_type};base64,{encoded_image}")
                    preview_src = f"data:{mime_type};base64,{encoded_image}"
                    st.session_state["image_bytes"].append(file_bytes)

                with st.spinner(f"Processing {file_name}..."):
                    try:
                        # ✅ FIX: use direct HTTP call instead of client.ocr.process()
                        ocr_response = call_mistral_ocr(api_key, document)
                        time.sleep(1)

                        pages = ocr_response.get("pages", [])
                        result_text = (
                            "\n\n".join(
                                page["markdown"]
                                for page in pages
                                if "markdown" in page
                            )
                            or "No result found."
                        )
                    except Exception as e:
                        result_text = f"Error extracting result: {e}"
                    st.session_state["ocr_result"].append(result_text)
                    st.session_state["preview_src"].append(preview_src)

# 5. Display Preview and OCR Results if available
if st.session_state["ocr_result"]:
    for idx, result in enumerate(
        st.session_state["ocr_result"]
    ):
        st.markdown(
            """
            <div style="
                background: rgba(255,255,255,0.06);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 20px;
                margin-bottom: 30px;
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            ">
            """,
            unsafe_allow_html=True
        )
        col1, col2 = st.columns([1, 1])
        # =====================================================
        # LEFT PANEL → PREVIEW
        # =====================================================
        with col1:
            st.markdown(
                f"""
                <h3 style="
                    color:#93c5fd;
                    font-weight:700;
                ">
                    📄 Input File {idx+1}
                </h3>
                """,
                unsafe_allow_html=True
            )
            if file_type == "PDF":
                pdf_embed_html = f"""
                <iframe
                    src="{st.session_state["preview_src"][idx]}"
                    width="100%"
                    height="850"
                    style="
                        border-radius:16px;
                        border:1px solid rgba(255,255,255,0.1);
                    "
                ></iframe>
                """
                st.markdown(
                    pdf_embed_html,
                    unsafe_allow_html=True
                )
            else:
                if (
                    source_type == "Local Upload"
                    and st.session_state["image_bytes"]
                ):
                    st.image(
                        st.session_state["image_bytes"][idx],
                        use_container_width=True
                    )
                else:
                    st.image(
                        st.session_state["preview_src"][idx],
                        use_container_width=True
                    )
        # =====================================================
        # RIGHT PANEL → OCR RESULT
        # =====================================================
        with col2:
            st.markdown(
                f"""
                <h3 style="
                    color:#c4b5fd;
                    font-weight:700;
                ">
                     OCR Result {idx+1}
                </h3>
                """,
                unsafe_allow_html=True
            )
            # ================= METRICS =================
            word_count = len(result.split())
            char_count = len(result)
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    "📝 Words",
                    word_count
                )
            with m2:
                st.metric(
                    "🔠 Characters",
                    char_count
                )
            st.markdown("<br>", unsafe_allow_html=True)

            # ================= TABS =================
            tab1, tab2, tab3 = st.tabs([
                "✨ Rendered",
                "📄 Raw Text",
                "🧾 JSON"
            ])
            # ================= RENDERED =================
            with tab1:
                st.markdown(
                    """
                    <style>
                    .scrollable-markdown {
                        max-height: 700px;
                        overflow-y: auto;
                        padding: 18px;
                        border-radius: 14px;
                        border: 1px solid rgba(255,255,255,0.08);
                    }
                    .scrollable-markdown::-webkit-scrollbar {
                        width: 8px;
                    }

                    .scrollable-markdown::-webkit-scrollbar-thumb {
                        background: rgba(255,255,255,0.2);
                        border-radius: 10px;
                    }

                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="scrollable-markdown">
                        {result}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # ================= RAW TEXT =================
            with tab2:
                st.text_area(
                    "Raw OCR Output",
                    result,
                    height=700
                )

            # ================= JSON =================
            with tab3:
                json_data = json.dumps(
                    {"ocr_result": result},
                    ensure_ascii=False,
                    indent=2
                )
                st.json(json.loads(json_data))
            # ================= DOWNLOAD SECTION =================
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <h4 style="
                    color:#c4b5fd;
                    font-weight:600;
                ">
                    Export Results
                </h4>
                """,
                unsafe_allow_html=True
            )
            d1, d2, d3 = st.columns(3)
            with d1:
                create_download_link(
                    json_data,
                    "application/json",
                    f"Output_{idx + 1}.json"
                )
            with d2:
                create_download_link(
                    result,
                    "text/plain",
                    f"Output_{idx + 1}.txt"
                )
            with d3:
                create_download_link(
                    result,
                    "text/markdown",
                    f"Output_{idx + 1}.md"
                )
        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )
