```shell
saved_model_cli run \
    --dir . \
    --tag_set serve \
    --signature_def serving_default \
    --input_exprs='user_id=[22]'
```

```shell
saved_model_cli show \
    --dir . \
    --tag_set serve \
    --signature_def serving_default
```