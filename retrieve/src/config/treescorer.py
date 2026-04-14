import pydantic
import yaml

from .base import EnvYaml


class DatasetYaml(pydantic.BaseModel):
    name: str
    text_encoder_name: str


class TreeScorerYaml(pydantic.BaseModel):
    max_hops: int
    max_paths_per_root: int = 200
    max_paths_per_sample: int = 1000
    add_reverse_edges: bool = True
    hidden_size: int
    num_layers: int = 2
    heads: int = 1
    max_neg_per_pos: int = 10
    max_neg_per_sample: int = 60
    use_cache: bool = True
    cache_version: str = "v3"


class OptimizerYaml(pydantic.BaseModel):
    lr: float


class EvalYaml(pydantic.BaseModel):
    k_list: str


class TreeScorerTrainYaml(pydantic.BaseModel):
    num_epochs: int
    patience: int
    batch_size: int
    save_prefix: str


class TreeScorerExpYaml(pydantic.BaseModel):
    env: EnvYaml
    dataset: DatasetYaml
    treescorer: TreeScorerYaml
    optimizer: OptimizerYaml
    eval: EvalYaml
    train: TreeScorerTrainYaml


def load_yaml(config_file):
    with open(config_file, encoding="utf-8") as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop("task")
    assert task == "treescorer"

    config = TreeScorerExpYaml(**yaml_data).model_dump()
    config["eval"]["k_list"] = [
        int(k) for k in config["eval"]["k_list"].split(",")
    ]

    return config
