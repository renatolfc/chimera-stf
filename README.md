# Chimera-STF

This repository implements the Chimera Shared Matrix Factorization (over Time)
technique, first introduced in the paper "Temporally Evolving Community Detection
and Prediction in Content-Centric Networks".

This algorithm can simultaneously account for graph links, content, and temporal
analysis by extracting the latent semantic structure of the network in
multidimensional form, but in a way that takes into account the temporal
continuity of these embeddings.

The code in this repo implements the loss function

![Loss function to be minimized](./doc/model.svg)

Once optimization converges or time runs out, it will save the learning
embeddings in their own files.

# Citing this work

If the code in this repository somehow helps your research, please consider
citing the aforementioned paper. A BibTeX entry is provided for you below:

```
@inproceedings{appel2018temporally,
  title={Temporally Evolving Community Detection and Prediction in COntent-Centric Networks},
  author={Ana P. Appel and Renato L. F. Cunha and Charu Aggarwal and Marcela Megumi Terakado},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={},
  month = {September},
  year={2018},
  organization={Springer},
  address = {Dublin, Ireland},
}
```
