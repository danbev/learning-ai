## AI/ML Investigation for Trusification
This task is about investigating the use of AI/ML for trusification and see if
there are ways to add value using AI/ML for trustification.

The suggestions in this document are just that, suggestions, and some might not
make sense or be feasible. This more of a brainstorming document.

### Suggestion: Vector database for VEX documents
Currently the trusification application/ui enables users to store SBOMs and
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
them. It is not using a vector database. But it could be a starting point for
experimenting with this idea and vector databases have more features like
filtering and such and might be worth looking into. By using a vector database
and having the information in it we could use this in combinarion with an LLM
to first query the vector database for documents relative to the query and then
pass them along to an LLM as context so that a task/query agains the LLM can
use that information. This was it would be possible for the LLM to "have
knowledge" about the vulnerabilities in the vector database. The same concept
is shown in [github-search.py](../../langchain/src/github-search.py) but in this case
it uses documents from github repository as the document source.

What is the benefit of this?   
The motivation would be that we might then be able to extract all the
dependencies of a project from whatever build system, or package information
exists and provide a list of vulnerabilities for the project.


### Suggestion: Measure of confidence of software projects
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

### Suggestion: Fine tune a language model to detect vulnerabilities
This is a suggestion to fine tune a language model to detect vulnerabilities
specific to Red Hat products. Or perhaps using something like RAG (Retrieval
Augmented Generation). After some more reading I don't think is a good idea
and if we wanted to use a language model we should use a vector database like
mentioned previously to query for similar vulnerabilities and the send them
along to the LLM as context so that the LLM has access to the information.

### Suggestion: Use a language model to generate a vulnerability report
Similar to the previous suggestion, but instead of detecting vulnerabilities
we could use a language model to generate a vulnerability report. This could
be done by using a language model to generate a report based on a VEX
information and other information that we have available to us in the
trustification database. If we could collect this information and add it as
context we could use an LLM to generate a report.
