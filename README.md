## Distinct Label Representations for Few-shot Text Classification

### Dataset
Dataset can be available on the [repo](https://github.com/YujiaBao/Distributional-Signatures) of Bao et al.


### Usage
```buildoutcfg
# For run specific setting
python3 train.py --data huffpost --data_dir data/ --config_files config.yaml --lr 1e-5 --alpha 1e-4 

# For searching hyperparameters
python3 train.py --data huffpost --data_dir data/ --config_files config.yaml --search_params

# For evaluating
python3 train.py --data huffpost --data_dir data/ --config_file config.yaml --eval_model_path /path/to/model --test_N 5 --test_K 1
```

###  Config.yaml
| name | description |
| ---- | ---- |
| model | The name of a model. Please set the value one of following: `["proto", "mlman", "maml"]`
| num_train_steps | The maximum number of training steps
| num_eval_steps | The number of evaluation steps |
| early_stop | The threshold for early stopping. If the model cannot beat the best socre for `early_stop` epochs, training is stopped. |
| eval_interval | a model will be evaluated every `eval_interval` steps |
| meta_setting["K"] | The number of examples for each label |
| meta_setting["N"] | The number of labels |
| meta_setting["Q"] | The number of elements in query set |



### Citation
```buildoutcfg
@inproceedings{ohashi-etal-2021-distinct,
    title = "Distinct Label Representations for Few-Shot Text Classification",
    author = "Ohashi, Sora  and
      Takayama, Junya  and
      Kajiwara, Tomoyuki  and
      Arase, Yuki",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.105",
    doi = "10.18653/v1/2021.acl-short.105",
    pages = "831--836",
    abstract = "Few-shot text classification aims to classify inputs whose label has only a few examples. Previous studies overlooked the semantic relevance between label representations. Therefore, they are easily confused by labels that are relevant. To address this problem, we propose a method that generates distinct label representations that embed information specific to each label. Our method is applicable to conventional few-shot classification models. Experimental results show that our method significantly improved the performance of few-shot text classification across models and datasets.",
}
```

