import streamlit as st
import os
import tempfile
from core.rag_handler import get_structured_rag_response
from core.processing import process_document_for_rag, LOADER_MAPPING

# --- 1. Page Configuration ---
st.set_page_config(page_title="Interactive Policy Engine", page_icon="🚀", layout="wide")

# --- 2. Caching the Vector Store Creation ---
@st.cache_resource
def get_vector_store_from_file(uploaded_file):
    """Processes an uploaded file and returns a cached vector store."""
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_filepath = tmp_file.name
            
            vector_store = process_document_for_rag(tmp_filepath)
            os.remove(tmp_filepath)
            return vector_store
        except Exception as e:
            st.error(f"Error processing document: {e}")
            return None
    return None

# --- 3. UI Layout ---
st.title("🚀 Interactive Document Decision Engine")
st.markdown("Upload a policy document to begin.")

# --- 4. Sidebar for Document Upload ---
with st.sidebar:
    st.header("Upload Document")
    supported_types = [key.lstrip('.') for key in LOADER_MAPPING.keys()]
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=supported_types,
        label_visibility="collapsed"
    )
    
    # --- THIS BUTTON IS THE FIX ---
    # We need a button to trigger the processing and caching.
    process_button = st.button("Process Document", type="primary")

# --- 5. Main Logic ---
vector_store = None
if process_button and uploaded_file:
    with st.spinner("Processing document... This may take a moment."):
        vector_store = get_vector_store_from_file(uploaded_file)
        if vector_store:
            st.session_state.doc_processed = True
            st.session_state.messages = [] # Clear chat on new doc
            # We use st.rerun() to smoothly update the UI after processing
            st.rerun() 
        else:
            st.error("Failed to process document.")
            st.session_state.doc_processed = False

# --- 6. Chat Interface ---
if st.session_state.get("doc_processed", False):
    st.header("Ask Your Questions")
    
    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Retrieve the vector store again for the chat logic
                # The cache ensures this is instantaneous
                current_vector_store = get_vector_store_from_file(uploaded_file)
                if current_vector_store:
                    response_dict = get_structured_rag_response(prompt, current_vector_store)
                    
                    if "error" in response_dict:
                        error_message = response_dict["error"]
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                    else:
                        decision = response_dict.get('decision', 'N/A')
                        amount = response_dict.get('amount', 0.0)
                        justification = response_dict.get('justification', 'No justification provided.')
                        clauses = response_dict.get('source_clauses', [])

                        response_md = f"""
                        **Decision:** `{decision}`\n
                        **Amount:** `₹{amount:,.2f}`\n
                        **Justification:** {justification}\n
                        """
                        st.markdown(response_md)

                        if clauses:
                            with st.expander("**Cited Policy Clauses**", expanded=True):
                                for clause in clauses:
                                    st.code(clause, language='text')

                        st.session_state.messages.append({"role": "assistant", "content": response_md})
                else:
                    st.error("Could not find the processed document. Please try processing it again.")
else:
    st.warning("Please upload a document and click 'Process Document' to start the analysis.")

