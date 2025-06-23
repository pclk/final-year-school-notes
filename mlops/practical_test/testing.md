Briefly describe TWO (2) components of MLFlow that help in managing the Machine Learning lifecycle?
The tracking server component tracks parameters, model metrics, metadata models, and artifacts. This helps with experimentation phase.
The model registry component stores, annotates and manages models in a central repository. This helps with deployment and operationalization phase.

Briefly describe TWO (2) benefits of using Hydra?
Switching different configuration groups easily.
Automated recording of execution results, along with code and configuration.

Experimentation takes place throughout the entire model development process. Briefly describe TWO(2) goals that we can achieve through experimentation?
Finding the best modelling parameters (algorithms, hyperparameters, feature processing, etc.)
Finding a balance between model improvement and computation costs, since there's always room for improvement, how good is good enough?

Depending on the impact of the model's predictions, decisions, or classifications, a more or less deep understanding may be required. Briefly describe TWO (2) reasonable steps data scientists should take to ensure the model is not actively harmful?
Checking the model with different sets of inputs, and looking for any oddities or extreme variance in the outputs.
Splitting a dimension and checking for different behaviour and metrics for different subpopulations. For example, is the error rate same for male and female?

MLOps as the process of automating machine learning using DevOps methodologies. Briefly describe TWO (2) best practices from DevOps that have been adopted in MLOps?
Continuous Integration is the process of continually testing the software project, and improving its quality based on the tests' results. Automated code testing is done with open source, SaaS, or cloud native build servers.
Continuous Delivery delivers code in new environments without human intervention. Automated code deployment is usually done with Infrastructure as Code (IaC)
