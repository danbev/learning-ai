## AI/ML Investigation for Trusification
This task is about investigating the use of AI/ML for trusification and see if
there are ways to add value add to using AI/ML for trusification.

### Suggestion: Vector database for VEX documents
Currently the trusification application/ui enables users to store SBOMs and
VEX (Vulnerability Exchange) documents in a database and then query the database
for vulnerabilities.

So in this case the user need to manually know about what vulnerabilities or
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
