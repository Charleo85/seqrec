## Model implementation for the CTA mechanism

the CTA mechanism is first proposed in the WWW 2020 proceeding: *Déjà vu: A Contextualized Temporal Attention Mechanism for Sequential Recommendation*  [arxiv](https://arxiv.org/abs/2002.00741)

BibTeX for citation:
```
@misc{wu2020dj,
    title={Déjà vu: A Contextualized Temporal Attention Mechanism for Sequential Recommendation},
    author={Jibang Wu and Renqin Cai and Hongning Wang},
    year={2020},
    eprint={2002.00741},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
```


For the dataset mentioned in the paper, you use sample script

```
./xing.sh 
./taobao.sh
```

Note:
`--kernel_type a-2-b-10` flag specifies the beta kernel configuration: 2 kernels of type a and 10 kernels type b 
