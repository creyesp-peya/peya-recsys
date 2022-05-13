```shell
saved_model_cli run \
    --dir /home/creyesp/DataspellProjects/mlp-project/data/dataset/models/index_bruteforce_model/ \
    --tag_set serve \
    --signature_def serving_default \
    --input_exprs='input_1=[22]'
```