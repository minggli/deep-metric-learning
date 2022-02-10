# Deep Metric Learning
This repository contains exploratory study of recent advancements of Deep Metric Learning in particular Contrastive Representation Learning.

InfoNCE is based on Noise Contrastive Estimation and aims to maximise Mutual Information of query-document pair (context and signal) which itself is a generalization of Triplet Loss to multiple negative examples.

Soft Nearest Neighbour Loss further generalises InfoNCE to allow multiple positive examples.

## Embedding
Visualisation were produced using representation of last layer of untrained `ResNet18` and t-SNE.

### MNIST (Cross Entropy, InfoNCE, Soft Nearest Neighbour Loss)
| Cross Entropy | InfoNCE      |Soft NN Loss|
|:-------------:|:------------:|:----------:|
|![alt text](assets/images_1644456003.gif "MNIST, CrossEntropyLoss")|![alt text](assets/images_1644456757.gif.gif "MNIST, InfoNCE")|![alt text](assets/images_1644457560.gif "MNIST, SoftNearestNeighbourLoss")|

### Fashion-MNIST (Cross Entropy, InfoNCE, Soft Nearest Neighbour Loss)


### Requirement
`poetry install`

