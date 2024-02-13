## Machine Learning Bill of Materials (ML-BOM)
Is a documentation of an algorithm's dataset and model, providing transparency
into possible security, privacy, safety, and ethical considerations. It is
designed to offer visibility into the training datasets and deployment methods
used behind machine learning models, aiming to increase transparency for all
stakeholders, from providers to consumers, resellers, and end-consumers. 

An ML-BOM contains a detailed list of all the components involved in the
`development` and `deployment` of a machine learning model. This includes:

* Sources of the data used to train the model. These not always available except
for a few open source data sets. This can be tricky as some opensource models
are trained on data that might have been generated using an other model. For
example one might use ChatGPT-4 to generate training data.

* The code and algorithms that defines the models architecture.

* The environment that was used to train the model (which might be important if
it could be compromised). This is also for reproducibility so others can
replicate the results (not really sure who could afford that but it would at
least enable someone do to so).

* The environment where the model is deployed.

### CycloneDX ML BOM
https://cyclonedx.org/capabilities/mlbom/
