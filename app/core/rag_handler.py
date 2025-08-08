import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

class PolicyDecision(BaseModel):
    decision: str = Field(description="The final decision, e.g., 'Approved', 'Rejected', 'Information Not Found'.")
    amount: float = Field(description="The approved claim amount. Return 0.0 if not applicable or rejected.")
    justification: str = Field(description="A clear, step-by-step explanation for the decision. If info is not found, state what information is missing.")
    source_clauses: list[str] = Field(description="A list of the exact, verbatim text of the policy clauses used. If no relevant clauses are found, return an empty list.")

def get_structured_rag_response(query: str, vector_store: Chroma) -> dict:
    if not query:
        return {"error": "Query cannot be empty."}
    if not vector_store:
        return {"error": "Vector store not found. Please process a document first."}

    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        parser = PydanticOutputParser(pydantic_object=PolicyDecision)

        # --- The Polished Prompt ---
        template = """
        **Your Role:** You are a meticulous and impartial Insurance Claims Adjudicator.

        **Your Prime Directive:** Your analysis and decision must be based **exclusively** on the provided `CONTEXT`. You must not use any prior knowledge or make assumptions.

        **Your Strict Rules:**
        1.  **No External Knowledge:** If the `CONTEXT` does not contain the information needed to answer the `QUERY`, you MUST state that the information is not found.
        2.  **No Hallucinations:** Do not invent details, figures, or clauses. Adhere strictly to the provided text.

        **The Adjudication Process (Follow these steps precisely):**
        1.  **Analyze the Query:** Deconstruct the user's `QUERY` to understand the specific claim being made.
        2.  **Scan for Evidence:** Scour the `CONTEXT` to find the most relevant policy clauses that apply to the `QUERY`.
        3.  **Formulate Decision:** Based *only* on the evidence from Step 2, decide if the claim is 'Approved', 'Rejected', or if key 'Information Not Found'.
        4.  **Construct Justification:** Write a clear, step-by-step explanation for your decision, referencing the logic from the clauses you found. If information was not found, state exactly what was missing.
        5.  **Source Verbatim:** Compile a list of the exact, word-for-word `source_clauses` from the `CONTEXT` that you used in your analysis. If no clauses were used, provide an empty list.
        6.  **Final Output:** Assemble your findings into the required JSON format.

        --- CONTEXT ---
        {context}
        --- END CONTEXT ---

        --- QUERY ---
        {question}
        --- END QUERY ---

        Adhere to the process and provide your response in the specified JSON format.
        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": query})
        parsed_output = parser.parse(result['result'])
        return parsed_output.model_dump()

    except Exception as e:
        print(f"Error in RAG response generation: {e}")
        return {
            "decision": "Error",
            "amount": 0.0,
            "justification": f"An internal error occurred: {str(e)}",
            "source_clauses": []
        }