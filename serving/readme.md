# Tensorflow serving

## Run in docker

### Base model
```
docker pull google/tf-serving-scann:2.6.0

MODEL_PATH="$(pwd)/models/base_line_model/index_model/"
MODEL_NAME=base_line
docker run -p 8501:8501 \
  --mount type=bind,source=${MODEL_PATH},target=/models/${MODEL_NAME} \
  -e MODEL_NAME=${MODEL_NAME} \
  -t google/tf-serving-scann:2.6.0
```
Request
```
curl -X POST http://localhost:8501/v1/models/${MODEL_NAME}:predict \
  --header "Content-Type: application/json" \
  --data '{"instances": ["2222"]}'

```
Response
```bash
{
    "predictions": [
        {
            "output_1": [0.436828494, 0.430209041, 0.417791814, 0.413429201, 0.405470788, 0.401971161, 0.398775518, 0.396291584, 0.39364633, 0.389002353],
            "output_2": ["7791337002012", "7790742172006", "7791337000926", "7791813421580", "7790742333605", "7790895010088", "7791813421917", "7790742172105", "7796989075803", "7791337001978"]
        }
    ]
}
```
Response
```bash

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
Request
```
curl -X POST http://localhost:8501/v1/models/${MODEL_NAME}:predict \
  --header "Content-Type: application/json" \
  --data '{"instances": [{"user_id": 2222, "dow": 1, "hod": 16}]}'
```
Response
```bash
{
    "predictions": [
        {
            "output_1": [288.934387, 288.933105, 288.909058, 288.899597, 288.877869, 288.870819, 288.865906, 288.862488, 288.848114, 288.83374],
            "output_2": ["7790250054726", "7790742340801", "7792540260138", "7790580956707", "7891528091716", "7794820014608", "7790670045199", "78924468", "7791290786394", "7790064000261"]
        }
    ]
}
```

## Documentation
### tfserving
* https://www.tensorflow.org/tfx/serving/

### Custom serving modes
* https://towardsdatascience.com/how-to-serve-different-model-versions-using-tensorflow-serving-de65312e58f7
* 