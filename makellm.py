from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import uuid
import chromadb
import pandas as pd
import json

# Initialize the ChatGroq LLM
llm = ChatGroq(
    model='llama3-8b-8192',
    temperature=0,
    groq_api_key='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'
)

# Load the job posting webpage
loader = WebBaseLoader("https://simplify.jobs/p/e5ec76ac-ef44-457c-a5df-2db158638b12/Software-Engineering-Simulation-Intern--2025-Summer-Intern")
pagedata = loader.load().pop().page_content

# Create a prompt template to extract job details in JSON format
prompt_extract = PromptTemplate.from_template(
    """
    ###SCRAPED TEXT FROM WEBSITE:
    {pagedata}
    ###INSTRUCTION
    The scraped text is from the career's page of a website.
    Your job is to extract the job postings and return them in a JSON format containing the following keys: 'role', 'experience', 'skills' and 'description'.
    Only return the valid JSON (NO PREAMBLE, NO EXTRA TEXT):
    """
)

# Chain the prompt template with the LLM for response extraction
chainextract = prompt_extract | llm
r = chainextract.invoke(input={'pagedata': pagedata})

# Clean up LLM response to extract JSON portion only
try:
    json_data = r.content.strip().split("```")[1]  # Extract JSON from the response
    json_r = json.loads(json_data)  # Parse JSON
except (IndexError, json.JSONDecodeError):
    raise ValueError("Failed to extract job posting details from LLM output.")

# Print the extracted job posting for confirmation
print(json_r)

# Assuming 'json_r' is a valid job posting (a dictionary)
job_posting = json_r

# Initialize the ChromaDB client and collection
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name='portfolio')

# Load the portfolio CSV file
df = pd.read_csv("my_portfolio.csv")

# Add portfolio data to the ChromaDB collection if it's not already present
if not collection.count():
    for _, row in df.iterrows():
        collection.add(
            documents=row["Techstack"],
            metadatas={"links": row["Links"]},
            ids=[str(uuid.uuid4())]
        )

# Extract 'skills' from the job_posting
skills = job_posting.get('skills', '')

# Query the collection using the extracted skills
if skills:
    links = collection.query(query_texts=[', '.join(skills)], n_results=2).get('metadatas', [])
else:
    print("No skills found in job posting, cannot query portfolio links.")
    links = []

# Now use 'job_posting' for email generation
prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}
    
    ### INSTRUCTION:
    You are Sahil Pardasani, an international student at Penn State University, currently pursuing a bachelor's degree in computer science with minors in math and engineering leadership development. Your skills include Python, Java, SQL, HTML, CSS, and Tableau. Your job is to write a cold email to the client regarding the job mentioned above describing your capability in fulfilling their needs.
    Remember, you are Sahil Pardasani, a Penn State University student. Do not provide a preamble.
    ### EMAIL (NO PREAMBLE):
    """
)

# Chain the email prompt template with the LLM
chain_email = prompt_email | llm
r = chain_email.invoke({
    "job_description": str(job_posting),  # Pass the job_posting dictionary as the job description
    "link_list": links  # Pass the relevant portfolio links
})

# Print the email content
print("Generated Email:")
print(r.content)
