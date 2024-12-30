PyTorch reimplementation of self-supervised DL algorithms that I frequently use. Changes have been made from the original papers as I see fit. Code is for personal use but might have some good insights into implementation details that weren't elaborated on in the papers (details which I spent hours combing the source code for).

#### Main Algorithms
- SimCLR: [T Chen et al.](https://doi.org/10.48550/arXiv.2002.05709), Feb. 2020
- MoCo:
    - V1: (ILY Kaiming) [He et al.](https://doi.org/10.48550/arXiv.1911.05722), Nov. 2019
    - V2: [X Chen et al.](https://doi.org/10.48550/arXiv.2003.04297), Mar. 2020
- BYOL: [Grill et al.](https://doi.org/10.48550/arXiv.2006.07733), Jun. 2020
#### TODO
- CLIP: [OpenAI](https://doi.org/10.48550/arXiv.2103.00020), Feb. 2021
- SwAV: [Caron et al.](https://doi.org/10.48550/arXiv.2006.09882), Jun. 2020


### dev plan
Each algo should interface in mostly the same way:
- Function: (dataset, hpp, config, and callbacks) -> (trained model)
    - User will be responsible for data loading. The user may provide a batch_fn that converts the output of the data loader to the format expected by the algorithm.
    - Hyperparameters (model-affecting parameters) include the base network, temperatures/importance weights, optimizer (+args).
    - Configuration includes any performance-invariant parameters (CUDA, verbose, etc.)
- Also add utils for validation
