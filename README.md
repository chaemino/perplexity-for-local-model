# perplexity-for-local-model
Calculation Perplexity  for local model

ref: https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py

- test data: `test_data.txt`

```
"lorem ipsum"
"Happy Birthday!"
"Bienvenue"
```

- command

```
python get_perplexity_for_mymodel.py --model_path=gpt2 -bs=16 -ml=128 --data_dir_path=data/ --test_data_file=test_data.jsonl
```

- result

```
646.72 ## mean perplexitiy
[32.25164031982422, 1499.6246337890625, 408.2724914550781]
```
