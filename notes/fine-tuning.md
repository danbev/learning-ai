## Fine-tuning an LLM
First fine-tuning is about taking a pre-trained model and training it on
specific domain knowledge, or a specific task. This will make the model more
accurate for that specific task. So instead of a general model we will have a
speciallized one.


### Pre-training (for the general/base model)
First data is required to be gathered for the training process, and it might
need to be pre-processed, for example by removing sensitive information,
correcting errors, etc.
We then take that data and tokenize it, which is just the process of splitting
the data into smaller units. These tokens are the feed into a transformer neural
network and the network is trained to predict the next token in the sequence and
will adjust the weights during this process.

Pre-training gives the model a general understanding of the language and
knowledge of the world. This is done by training on a large amount of data
and is a very expensive process.

### Fine-tuning
This is the process of taking the pre-trained model and training it on a
specific task, or specific domain knowledge that might not have been present (
or not enough) in the training input data. This is where things like making the
model be an expert in answering questions (chat), or generating code.
We need to make sure that the data we use here is also pre-processed similar to
what was done for the pre-training data.

### Identifying key concepts

*What are the key concepts that we want to teach the model?*   
So if we want our trained model to know about things like VEX documents, SBOM,
and CVEs then we should gather data/documents that discuss these concepts in
context. We might need to create sentences ourselves that clearly explain these
concepts and examples.

VEX Document (Vulnerability Exploitability Exchange) example:
```
"When we received the VEX document from our software vendor, it detailed how the
recently discovered vulnerabilities affected their products. The VEX document
was crucial for our risk assessment, as it clearly outlined which
vulnerabilities were not applicable to our current system configuration,
allowing us to prioritize our patching strategy effectively."
```

SBOM (Software Bill of Materials) example:
```
"Our organization mandates the submission of a detailed SBOM from all software
suppliers. An SBOM is a comprehensive inventory of all components, libraries,
and modules used in the software. By analyzing the SBOM, we can swiftly identify
if any component is affected by a known vulnerability, enhancing our supply
chain security and reducing the risk of introducing vulnerable elements into our
systems."
```


CVE (Common Vulnerabilities and Exposures) example:
```
"During our security review, we cross-referenced the components listed in the
SBOM with the CVE database. The Common Vulnerabilities and Exposures (CVE)
system provides a public list of disclosed cybersecurity vulnerabilities. By
doing this, we discovered that one of the components had an associated
CVE-2021-XXXX, which was a critical vulnerability, leading us to immediately
contact the vendor for an update."
```

Integration of VEX and CVE in Supply Chain Security example:
```
"In our supply chain security protocol, we integrate information from VEX
documents with data from the CVE listings. This approach helps us understand not
only the existence of vulnerabilities in our software components (as listed in
CVEs) but also their exploitability in our specific environment (as detailed in
VEX documents). This dual-analysis method significantly improves our
decision-making process regarding software updates and patches."
```

SBOM for Compliance and Risk Management example:
```
"As part of our compliance with industry cybersecurity standards, we maintain
an up-to-date SBOM for all critical software used in our operations. The SBOM
aids us not only in supply chain security but also in risk management. By having
a detailed view of the software components, we can quickly assess our exposure
when new vulnerabilities are reported and ensure that we are not in violation
of regulatory requirements."
```

If we want our funetuned model to be able to answer questions we should create
questions-and-answer pairs that we think that users might ask or that we think
are important to know.
