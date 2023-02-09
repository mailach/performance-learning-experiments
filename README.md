# Performance Learning Experimentation

# Setting up the project locally
You need a Docker, Git, and conda installation (at least if you want to load systems).   
Clone the project with all submodules:


```sh
git clone git@github.com:mailach/performance-learning-experiments.git
git submodule init
git submodule update
```

Generate a conda environment from `conda.yml`.
You need to create an environment file, corresponding to the `.env.example`. If you want to try the project on a remote server, you can use my mlflow test server, ask me for credentials. 
Load your `.env` file:

```sh
export $(grep -v '^#' .env | xargs)
```

## Example experiments

You can use the provided examples to learn about how to use the executor. If you want to use the provided YAML files, you need to edit them so the data dir points to the absolute path of the executor. 


## Related repositories

### Server setup
* <https://github.com/mailach/mlflow-with-proxied-artifact-storage>

### Integrations
* SPLConqueror <https://github.com/mailach/SPLC-mlflow>
* DeepPerf <https://github.com/mailach/deepperf-mlflow>
* DECART <https://github.com/mailach/decart-mlflow>

### SPLC2py
* <https://github.com/mailach/SPLC2py>
