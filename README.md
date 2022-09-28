# Master's Thesis

Load .env file

```sh
export $(grep -v '^#' .env | xargs)
```

Run multistep workflow in root dir

```sh
mlflow run . -P test=x
```
