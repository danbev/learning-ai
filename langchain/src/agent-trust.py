import os, json

from langchain.requests import RequestsWrapper
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents import load_tools, Tool, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.agents.agent_toolkits.openapi import planner
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 
import requests

load_dotenv()

token_endpoint = 'http://localhost:8090/realms/chicken/protocol/openid-connect/token'
client_id = 'walker'
client_secret = os.getenv("CLIENT_SECRET")
access_token = []

response = requests.post(
    token_endpoint,
    data={
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    }
)
if response.status_code == 200:
    token_data = response.json()
    access_token = token_data['access_token']
else:
    print(f"Failed to obtain access token. Status code: {response.status_code}, Response: {response.text}")
    exit()

# Download the OpenAPI spec for the API
# $ wget https://sbom.trustification.dev/openapi.json
# or locally
# $ wget http://localhost:8081/openapi.json
# 
# I needed to add the following element to the openapi spec:
# "servers": [{"url": "http://localhost:6030"}]}

with open("openapi.json") as f:
    raw_trust_api_spec = json.load(f)

def auth_headers(raw_spec: dict):
    scopes = list(
        raw_spec["components"]["securitySchemes"]["oidc"]["flows"][
            "authorizationCode"
        ]["scopes"].keys()
    )
    return {"Authorization": f"Bearer {access_token}"}

trust_api_spec = reduce_openapi_spec(raw_trust_api_spec)

headers = auth_headers(raw_trust_api_spec)
requests_wrapper = RequestsWrapper(headers=headers)

endpoints = [
    (route, operation)
    for route, operations in raw_trust_api_spec["paths"].items()
    for operation in operations
    if operation in ["get", "post"]
]
print(f'Server URL: {trust_api_spec.servers[0]["url"]}')
print("Endpoints:")
for endpoint in endpoints:
    print(endpoint)

trust_api_spec.endpoints.pop(1)
print(trust_api_spec.endpoints[0])

llm = OpenAI(model_name="gpt-4", temperature=0.0)

trust_agent = planner.create_openapi_agent(trust_api_spec,
                                           requests_wrapper,
                                           llm, 
                                           verbose=True,
                                           agent_executor_kwargs={
                                               'max_iterations': 5,
                                           })

prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
)
trust_agent.tools.append(llm_tool)
trust_agent.tools.append(load_tools(["google-serper"], llm=llm)[0])

#print(trust_agent.agent.llm_chain.prompt.template)
#print(trust_agent.tools)
#exit()
#print(trust_agent.agent.tools)
#exit()

result = trust_agent.run("Can you find the VEX for RHSA-2023:1441 and show me the CVEs?")
print(result)
