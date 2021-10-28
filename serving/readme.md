# Tensorflow serving

## Run in docker

### Base model
```
docker pull tensorflow/serving:2.6.0

MODEL_PATH="$(pwd)/models/base_line_index/"
MODEL_NAME=base_line
docker run -p 8501:8501 \
  --mount type=bind,source=${MODEL_PATH},target=/models/${MODEL_NAME} \
  -e MODEL_NAME=${MODEL_NAME} \
  -t tensorflow/serving
```

```
curl -X POST http://localhost:8501/v1/models/${MODEL_NAME}:predict \
  --header "Content-Type: application/json" \
  --data '{"instances": ["2222"]}'

```
### Hybrid model
```
docker pull google/tf-serving-scann:2.6.0


MODEL_PATH="$(pwd)/models/context_simple_index/"
MODEL_NAME=context_model
docker run -p 8501:8501 \
  --mount type=bind,source=${MODEL_PATH},target=/models/${MODEL_NAME} \
  -e MODEL_NAME=${MODEL_NAME} \
  -t google/tf-serving-scann:2.6.0
```

```
curl -X POST http://localhost:8501/v1/models/${MODEL_NAME}:predict \
  --header "Content-Type: application/json" \
  --data '{"instances": [{"user_id": 2222, "dow": 1, "hod": 16}]}'

```

## Documentation
### tfserving
* https://www.tensorflow.org/tfx/serving/

### Custom serving modes
* https://towardsdatascience.com/how-to-serve-different-model-versions-using-tensorflow-serving-de65312e58f7
* 