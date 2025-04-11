from datasets import load_dataset
from pprint import pprint

ds = load_dataset("keremberke/plane-detection", name="full")
example = ds['train'][0]

pprint(example)
