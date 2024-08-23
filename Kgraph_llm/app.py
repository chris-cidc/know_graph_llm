import pandas as pd
import json
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Neo4jVector
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

load_dotenv()




# Gemini (https://aistudio.google.com/app/apikey)
gemini_api = os.getenv("GEMINI_API")

# Hugging Face (if we want to use open source LLM)
hf_api = os.getenv("HF_API")

# Neo4j 
neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
print(neo4j_url)
neo4j_user = os.getenv("NEO4J_USER")
print(neo4j_user)
neo4j_password = os.getenv("NEO4J_PASSWORD")

# https://api.python.langchain.com/en/latest/graphs/langchain_community.graphs.neo4j_graph.Neo4jGraph.html
graph = Neo4jGraph(neo4j_url,neo4j_user,neo4j_password)


df = pd.read_csv('LinkedIn people profiles datasets.csv') 
df.head()


def extract_industry(json_str):
    try:
        data = json.loads(json_str)
        return data.get('industry', None)
    except json.JSONDecodeError:
        return None

def extract_languages(json_list):
    try:
        languages = [entry['title'] for entry in json.loads(json_list)]
        return '|'.join(languages)
    except: 
        return None

def extract_country(string):
    if isinstance(string, str):
        elements = string.split(',')
        return elements[-1].strip()  
    else:
        return None

df['industry'] = df['current_company'].apply(lambda x: extract_industry(x))
df['languages'] = df['languages'].apply(lambda x: extract_languages(x))
df['country'] = df['city'].apply(lambda x: extract_country(x))
df = df [['id','name','current_company:name','educations_details','languages','industry','country']].dropna()
industry_counts = df['industry'].value_counts()
df = df[df['industry'].isin(industry_counts[industry_counts > 2].index)].reset_index(drop=True)
df = df.rename(columns={'current_company:name': 'company','educations_details':'education'})
df.head(10)

# Insert data from csv to neo4j
graph.refresh_schema()
print(graph.schema)

people_query = """
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/chris-cidc/test/main/clean_data.csv'
AS row
MERGE (person:Person {name: row.name})
MERGE (company:Company {name: row.company})
MERGE (school:School {name: row.education})
MERGE (industry:Industry {name: row.industry})
MERGE (country:Country {name: row.country})

FOREACH (lang in split(row.languages, '|') | 
    MERGE (language:Language {name:trim(lang)})
    MERGE (person)-[:SPEAKS]->(language))

MERGE (person)-[:WORKS_IN]->(company)
MERGE (person)-[:LIVES_IN]->(country)
MERGE (person)-[:EDUCATED_AT]->(school)
MERGE (company)-[:IS_IN]->(industry)
"""

graph.query(people_query)


examples= [
    {
        "question": "Which workers speak French?",
        "query": "MATCH (p:Person)-[:SPEAKS]->(l:Language {{name: 'French'}}) RETURN p.name",
    },
    {
        "question": "What industries are workers named Emily associated with?",
        "query": "MATCH (p:Person {{name: 'Emily'}})-[:WORKS_IN]->(c:Company)-[:IS_IN]->(i:Industry) RETURN i.name",
    },
    {
        "question": "Which workers live in Canada and speak German?",
        "query": "MATCH (p:Person)-[:LIVES_IN]->(:Country {{name: 'Canada'}}), (p)-[:SPEAKS]->(:Language {{name: 'German'}}) RETURN p.name",
    },
    {
        "question": "In which countries do workers who speak Spanish live?",
        "query": "MATCH (p:Person)-[:SPEAKS]->(:Language {{name: 'Spanish'}})<-[:SPEAKS]-(worker:Person)-[:LIVES_IN]->(c:Country) RETURN DISTINCT c.name AS Country",
    },
    {
        "question": "What companies do workers named John work in?",
        "query": "MATCH (p:Person {{name: 'John'}})-[:WORKS_IN]->(c:Company) RETURN c.name",
    },
    {
        "question":"How many workers in Hospital and Health Care industry able to speak Korea",
        "query": "MATCH (p:Person)-[:WORKS_IN]->(:Company)-[:IS_IN]->(:Industry {{name: 'Hospitals and Health Care'}}),(p)-[:SPEAKS]->(:Language {{name: 'Korean'}}) RETURN COUNT(DISTINCT p) AS NumberOfWorkers",
    },
    {
        "question": "What companies are located in the technology industry?",
        "query": "MATCH (c:Company)-[:IS_IN]->(:Industry {{name: 'Technology'}}) RETURN c.name",
    },
    {
        "question": "Where do workers named Alice live?",
        "query": "MATCH (p:Person {{name: 'Alice'}})-[:LIVES_IN]->(c:Country) RETURN c.name",
    },
]

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = gemini_api ,temperature = 0)

example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}"
)
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    HuggingFaceEmbeddings(),
    Neo4jVector,
    url = neo4j_url,
    username = neo4j_user,
    password = neo4j_password,
    k=3,
    input_keys=["question"],
)


example_selector.select_examples({"question": "Where do Michael work?"})

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nBelow are a number of examples of questions and their corresponding Cypher queries.",
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question", "schema"],
)


chain2 = GraphCypherQAChain.from_llm(graph=graph, llm=llm, cypher_prompt=dynamic_prompt, verbose=True)

questions = ["List all companies in advertising services industry!",
             "A worker who graduated from Simon Fraser University is currently employed at?",
             "Where is Paul Lukes working?",
             "A worker residing in Canada who is proficient in Vietnamese?",
             "How many worker in United States speak Urdu?"]
for q in questions:
    print('====== START ======')
    print(chain2.invoke(q)['result'])
    print('====== END ====== \n')


print(dynamic_prompt.format(question="Where do Michael work?", schema="foo"))



