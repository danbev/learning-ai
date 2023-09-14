## AI/ML Investigation for Trustification
This task is about investigating the use of AI/ML for trustification and see if
there are ways to add value using AI/ML for trustification.

The suggestions in this document are just that, suggestions, and some might not
make sense or be feasible. This more of a brainstorming document.

### Vector database for VEX documents
Currently the trustification application/ui enables users to store SBOMs and
VEX (Vulnerability Exchange) documents in a database and then query the database
for vulnerabilities.

So in this case the user needs to manually know about what vulnerabilities or
what packages they use in their application to be able to query the database.

What if we took the VEX documents, or parts of them, and stored then in a vector
database:
```
                         Vector space
       +-----------------------------------------------+ 
       |                                 v₁            |
       |                                               |
       |        V₂                                     |
       |                 V₃                            |
       |                                               |
       |                                               |
       |                                               |
       +-----------------------------------------------+ 
                                            
                                           
V₁ = VEX for CVE-1
Vₛ = VEX for CVE-2
V₃ = VEX for CVE-3
```
And we could then take the package name and version of a project dependency and
create vector embeddings for them and use that to search for VEX related (near):
```
                         Vector space
       +-----------------------------------------------+ 
       |                                 v₁            |
       |                                  p₁           |
       |                                  ↑            |
       |                                  |            |
       |                                  |            |
       |                                  |            |
       |                                  |            |
       +----------------------------------|------------+ 
                                          | 
                                          |
V₁ = VEX for CVE-1234                     |
p₁ = Search for product name and version: +
```
Searching for vulnerabilities would then be a matter of finding the closest
vector in the vector database to the vector of the package name and version.
Just as an example to experiment with this idea there is
[vex.py](../../embeddings/python/src/vex.py). 

This example is just creating embeddings and computing the distance between
them. It does not use a vector database. But it could be a starting point for
experimenting with this idea and vector databases have more features like
filtering and such and might be worth looking into.

By using a vector database and having the information in it we could use this in
combination with an LLM to first query the vector database for documents
relative to the query and then pass them along to an LLM as context so that a
task/query agains the LLM can use that information. This way it would be
possible for the LLM to "have knowledge" about the vulnerabilities in our vector
database which it would not have access to otherwise. This concept is shown in
[github-search.py](../../langchain/src/github-search.py) but in this case
the example uses documents from github repository as the document source.
I believe that term for this model is
[Retrieval Augmented Generation (RAG)](./rag.md).

What is the benefit of this?   
The motivation would be that we might then be able to extract all the
dependencies of a project from whatever build system, or package information
exists and provide a list of vulnerabilities for the project.

By using a vector database it can be updated with new VEX documents as they
become available.

### Use a language model to generate a vulnerability report/summary
Similar to the previous suggestion, but instead of detecting vulnerabilities
we could use a language model to generate a vulnerability report. 

There can be a lot of information in a VEX document, and in the documents that
it references. What might be possible is to fetch this related data and pass
that to an LLM which can then generate a summary. This summary could then 
be displayed in the trustification UI.

This report could be provided in the users preferred language as well by
instructing the LLM to do so (would require the llm to have been trained on that
language I think).

As an initial investigation into this,
[vex-search.py](../../langchain/src/vex-search.py) takes a single VEX documents
and inserts it into an in-memory vector database. It will then use LangChain
to ask an LLM to generate a summary for the vulnerability using a question like
this:
```
"Summaries RHSA-2020:5566 using a short sentence, including a list of CVE's and references."
```
LangChain will first query the vector database for related documents and then
pass those to the LLM as context. The LLM will then generate a summary for the
vulnerability:
```console
(langch) $ python src/vex-search.py 
splits len: 8, type: <class 'langchain.schema.document.Document'>
Answer:
RHSA-2020:5566 is a Red Hat Security Advisory that provides a security update for OpenSSL in Red Hat Enterprise Linux 7. The update is rated as having an important security impact. The advisory includes a Common Vulnerability Scoring System (CVSS) base score for each vulnerability, which can be found in the References section. 

CVEs: 
- CVE-2020-1971
- CVE-2020-1970
- CVE-2020-1968

References:
- Red Hat Security Advisory: https://access.redhat.com/security/data/csaf/v2/advisories/2020/rhsa-2020_5566.json
- Red Hat Errata: https://access.redhat.com/errata/RHSA-2020:5566
- Red Hat Security Updates Classification: https://access.redhat.com/security/updates/classification/#important
- Red Hat Product Security Contact Details: https://access.redhat.com/security/team/contact/
```

This will also report the `source_documents` that were used to generate the summary:
```console
Source documents:
/home/danielbevenius/work/ai/learning-ai/langchain/src/vex-stripped.json
/home/danielbevenius/work/ai/learning-ai/langchain/src/vex-stripped.json
/home/danielbevenius/work/ai/learning-ai/langchain/src/vex-stripped.json
/home/danielbevenius/work/ai/learning-ai/langchain/src/vex-stripped.json
```

So that was a single VEX document, now how about we add some of the CVEs that
are referenced in that VEX to the vector store and see if we can generate a
summary for them:
```console
$ curl -s https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=CVE-2020-1971 > src/cve-2020-1971
$ curl -s https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=CVE-2020-1968 > src/cve-2020-1968
```
I tried adding both of these types of documents to the same vector store but
the results were not good and I'm not sure why. So I created separate vector
store for the CVEs. The process was then to chain the queries by first
performing a query for the VEX document, and then using the CVE that it refers
to to perform a query for the CVE information. This is shown in
[vex-cve.py](../../langchain/src/vex-cve.py).
```console
(langch) $ python src/vex_cve.py 
query='Show a short summary of RHSA-2020:5566, including the cve.'
result["answer"]=' RHSA-2020:5566 is an update for openssl for Red Hat Enterprise Linux 7. It has a security impact of Important and is related to CVE-2020-1971.'

query='Which CVEs were mentioned'
result["answer"]=' CVE-2020-1971'

formatted='Show me a detailed description of  CVE-2020-1971.'
result["answer"]=' CVE-2020-1971 is a NULL pointer de-reference vulnerability in OpenSSL, a toolkit that implements the Secure Sockets Layer (SSL) and Transport Layer Security (TLS) protocols, as well as a full-strength general-purpose cryptography library.'
```
I'm using OpenAI for these examples which means that they require an API key.
You can sign up and get a free trail period with token credits. The
[pricing](https://openai.com/pricing#language-models) is based on the number
of tokens used and it is possible to buy credits. I bought $10 worth of credits
and I'm still using not used up half of them yet.


### Use a language model to generate suggestions fixing vulnerabilities
The idea here would be that we gather information about the vulnerability
and craft a prompt to get an LLM to generate suggestions for how to fix the
reported vulnerability.

### Measure of confidence of software projects
* Recent github activity
* Release number >= 1.0
* Number of github stars?
* Signed commits?
* End of support date validation
* End of life date validation
* [OpenSSF Scorecard](https://securityscorecards.dev/#what-is-openssf-scorecard)

This might be possible to do by first collecting this information from a 
github repository and then creating a prompt which can classify this as
a secure project.

So we would create a prompt template, something like this:
```
Categorise the following project based on the
number of stars, and the number of commits. The more starts and
commits the better the project is likely to be.

stars: {stars}
commits: {commits}
```
We could then populate it with the information we have collected, and then
and ask an LLM to classify it as a secure project or not.  As an example
[project-health.py](../../langchain/src/project-health.py) is something along
these lines.
```console
(langch) $ python src/project-health.py 
https://api.github.com/repos/danbev/learning-v8 has 2300 stars and 30 commits

"Going to prompt chatgpt (gpt-3.5-turbo-0301)"
content='This project has a high number of stars but a low number of commits, which suggests that it may have gained popularity but may not be actively maintained or developed.' additional_kwargs={} example=False
```

As an example of the OpenSSF scorecard, I ran this against the
trustification repository:
<details>
<summary> OpenSSF scorecard </summary>

```console
$ /home/danielbevenius/go/scorecard --repo=github.com/trustification/trustification
Starting [Pinned-Dependencies]
Starting [Dangerous-Workflow]
Starting [License]
Starting [Fuzzing]
Starting [Security-Policy]
Starting [Vulnerabilities]
Starting [Code-Review]
Starting [Branch-Protection]
Starting [Token-Permissions]
Starting [Dependency-Update-Tool]
Starting [Binary-Artifacts]
Starting [Contributors]
Starting [SAST]
Starting [Packaging]
Starting [CI-Tests]
Starting [CII-Best-Practices]
Starting [Maintained]
Starting [Signed-Releases]
Finished [Security-Policy]
Finished [Vulnerabilities]
Finished [Code-Review]
Finished [Branch-Protection]
Finished [Token-Permissions]
Finished [Dependency-Update-Tool]
Finished [Binary-Artifacts]
Finished [Contributors]
Finished [SAST]
Finished [Packaging]
Finished [CI-Tests]
Finished [CII-Best-Practices]
Finished [Maintained]
Finished [Signed-Releases]
Finished [Pinned-Dependencies]
Finished [Dangerous-Workflow]
Finished [License]
Finished [Fuzzing]

RESULTS
-------
Aggregate score: 4.7 / 10

Check scores:
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
|  SCORE  |          NAME          |             REASON             |                                               DOCUMENTATION/REMEDIATION                                               |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | Binary-Artifacts       | no binaries found in the repo  | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#binary-artifacts       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 2 / 10  | Branch-Protection      | branch protection is not       | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#branch-protection      |
|         |                        | maximal on development and all |                                                                                                                       |
|         |                        | release branches               |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | CI-Tests               | 14 out of 14 merged PRs        | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#ci-tests               |
|         |                        | checked by a CI test -- score  |                                                                                                                       |
|         |                        | normalized to 10               |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | CII-Best-Practices     | no effort to earn an OpenSSF   | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#cii-best-practices     |
|         |                        | best practices badge detected  |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 7 / 10  | Code-Review            | found 4 unreviewed changesets  | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#code-review            |
|         |                        | out of 16 -- score normalized  |                                                                                                                       |
|         |                        | to 7                           |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | Contributors           | 25 different organizations     | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#contributors           |
|         |                        | found -- score normalized to   |                                                                                                                       |
|         |                        | 10                             |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | Dangerous-Workflow     | no dangerous workflow patterns | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#dangerous-workflow     |
|         |                        | detected                       |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | Dependency-Update-Tool | no update tool detected        | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#dependency-update-tool |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | Fuzzing                | project is not fuzzed          | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#fuzzing                |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | License                | license file detected          | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#license                |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 10 / 10 | Maintained             | 30 commit(s) out of 30 and 30  | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#maintained             |
|         |                        | issue activity out of 30 found |                                                                                                                       |
|         |                        | in the last 90 days -- score   |                                                                                                                       |
|         |                        | normalized to 10               |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| ?       | Packaging              | no published package detected  | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#packaging              |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 3 / 10  | Pinned-Dependencies    | dependency not pinned by hash  | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#pinned-dependencies    |
|         |                        | detected -- score normalized   |                                                                                                                       |
|         |                        | to 3                           |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | SAST                   | SAST tool is not run on all    | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#sast                   |
|         |                        | commits -- score normalized to |                                                                                                                       |
|         |                        | 0                              |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | Security-Policy        | security policy file not       | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#security-policy        |
|         |                        | detected                       |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 8 / 10  | Signed-Releases        | 5 out of 5 artifacts are       | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#signed-releases        |
|         |                        | signed or have provenance      |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | Token-Permissions      | detected GitHub workflow       | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#token-permissions      |
|         |                        | tokens with excessive          |                                                                                                                       |
|         |                        | permissions                    |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 0 / 10  | Vulnerabilities        | 17 existing vulnerabilities    | https://github.com/ossf/scorecard/blob/7ed886f1bd917d19cb9d6ce6c10e80e81fa31c39/docs/checks.md#vulnerabilities        |
|         |                        | detected                       |                                                                                                                       |
|---------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
```
</details>
