import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma

# --- 1. Load Environment Variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# --- 2. Define the desired JSON output structure ---
class PolicyDecision(BaseModel):
    decision: str = Field(description="The final decision, e.g., 'Approved', 'Rejected', 'Needs More Information'.")
    amount: float = Field(description="The approved claim amount. Return 0.0 if not applicable or rejected.")
    justification: str = Field(description="A clear, step-by-step explanation for the decision, including any calculations performed.")
    source_clauses: list[str] = Field(description="A list of the exact, verbatim text of the policy clauses used for the justification.")

# --- 3. The Advanced RAG Function with the Improved Prompt ---
def get_structured_rag_response(query: str, vector_store: Chroma) -> dict:
    if not query:
        return {"error": "Query cannot be empty."}
    if not vector_store:
        return {"error": "Vector store not found. Please process a document first."}

    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        parser = PydanticOutputParser(pydantic_object=PolicyDecision)

        # --- THIS IS THE CRUCIAL UPDATE ---
        template = """
        You are an expert insurance claims processor. Your role is to analyze a user's query and the relevant policy clauses to make a final, structured decision.

        Carefully review the following policy clauses retrieved based on the user's query:
        --- CONTEXT ---
        {context}
        --- END CONTEXT ---

        Now, analyze the following user query:
        --- QUERY ---
        {question}
        --- END QUERY ---

        Follow these steps precisely to make your decision:
        1.  **Identify Key Figures:** Extract the 'Sum Insured', the 'Property Value', and the 'Amount of Loss' from the query and context.
        2.  **Check for Underinsurance:** Determine if the 'Sum Insured' is less than the 'Property Value'.
        3.  **Evaluate Underinsurance Clauses:** There are two main underinsurance conditions. They are mutually exclusive. Evaluate them in this specific order:
            a.  **Proportionate Clause:** First, check if the 'Sum Insured' is less than 85% of the 'Property Value'. If it is, the claim must be reduced proportionately using the formula: (Sum Insured / Property Value) * Amount of Loss. Do not consider any other waivers.
            b.  **Waiver Clause:** ONLY if the condition in step 3a is NOT met, then check if the underinsurance is 15% or less. If it is, the underinsurance is waived, and the full 'Amount of Loss' is payable (up to the 'Sum Insured').
        4.  **Formulate Justification:** Explain which clause was applied and show the calculation you performed.
        5.  **Finalize Decision:** Based on your analysis, provide the final decision and the calculated payable amount.

        You must provide your response in a JSON format. Do not add any other text before or after the JSON.
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
            chain_type_kwargs={"prompt": prompt}
        )

        llm_output_str = qa_chain.invoke({"query": query})['result']
        parsed_output = parser.parse(llm_output_str)
        return parsed_output.model_dump()

    except Exception as e:
        print(f"Error in RAG response generation: {e}")
        return {"error": str(e)}
