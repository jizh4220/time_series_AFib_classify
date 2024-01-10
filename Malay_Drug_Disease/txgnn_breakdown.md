Here is a more detailed step-by-step summary of the Predictor and Explainer modules in TXGNN:

TXGNN Predictor:
- It uses a graph neural network (GNN) to learn representations of entities (drugs and diseases) in the biomedical knowledge graph. 
- The GNN applies several message passing layers to aggregate information from neighboring nodes in the graph.
- After pre-training, the GNN embeddings capture meaningful biological properties of entities.
- A metric learning loss is then used to fine-tune the GNN to predict relationships (indications, contraindications) between drugs and diseases.
- To enable zero-shot prediction, TXGNN constructs disease signature vectors based on neighbors in the knowledge graph. 
- Disease similarity is calculated as the normalized dot product of signature vectors. Similar diseases (>0.2 score) likely share mechanisms.
- For a query disease, TXGNN retrieves similar diseases, generates their embeddings, and aggregates them weighted by similarity.  
- The aggregated embedding transfers knowledge between related diseases to enable zero-shot prediction for those with no known drugs.
- TXGNN ranks drugs based on predicted likelihoods to output therapeutic candidates for the query disease.

TXGNN Explainer:
- It employs GraphMask, a self-explaining graph neural network method, to generate explanations.
- GraphMask identifies a sparse subgraph of entities critical to each drug-disease prediction.
- It assigns an importance score from 0-1 to each edge, quantifying relevance to the prediction.
- TXGNN Explainer combines the subgraph and scores to extract multi-hop paths connecting drugs to diseases. 
- These multi-hop explanations align with clinicians' intuitive ways of reasoning about repurposing hypotheses.
- A user interface visualizes explanations to assist clinicians in validating predictions.

In summary, the Predictor leverages knowledge graph embeddings while the Explainer provides interpretability for the model's therapeutic predictions through multi-hop explanatory paths.