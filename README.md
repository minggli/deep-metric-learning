# Deep Metric Learning
This repository contains exploratory study of recent advancements of Deep Metric Learning in particular Contrastive Representation Learning.

InfoNCE ([Oord et al 2019][1]) is based on Noise Contrastive Estimation (NCE [Gutmann and Hyvarinen][2]) and aims to maximise _Mutual Information_ of query-document pair (or as put originally, context and signal). NCE itself is a generalization of triplet loss used in FaceNet ([Schroff et al 2015][3]) to multiple negative examples.

Some intimate connection between NCE and Generative Adverserial Networks (GANs) have been discussed ([Goodfellow 2015][4], [Frosst 2019][5]).

Soft Nearest Neighbour Loss ([Frosst 2019][5]) further generalises InfoNCE to allow multiple positive examples.

## Embedding
Visualisation were produced using representation of last layer of untrained `ResNet18` and t-SNE.

### MNIST (Cross Entropy, InfoNCE, Soft Nearest Neighbour Loss)
| Cross Entropy | InfoNCE      |Soft NN Loss|
|:-------------:|:------------:|:----------:|
|![alt text](assets/images_1644456003.gif "MNIST, CrossEntropyLoss")|![alt text](assets/images_1644456757.gif "MNIST, InfoNCE")|![alt text](assets/images_1644457560.gif "MNIST, SoftNearestNeighbourLoss")|

### Fashion-MNIST (Cross Entropy, InfoNCE, Soft Nearest Neighbour Loss)


### Requirement
`poetry install`


[1]: https://arxiv.org/abs/1807.03748 
[2]: https://www.jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf
[3]: https://arxiv.org/abs/1503.03832
[4]: https://arxiv.org/abs/1412.6515
[5]: https://arxiv.org/abs/1902.01889
