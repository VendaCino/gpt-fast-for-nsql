# gpt-fast-nsql
This project draws inspiration from [gpt-fast](https://github.com/pytorch-labs/gpt-fast) and applies the same performance optimization strategy to nsql models.

[NSQL](https://github.com/NumbersStationAI/NSQL) is a family of autoregressive open-source large foundation models (FMs) designed specifically for SQL generation tasks.

Not prepared for production, only for learning. Please copy-paste and fork as you desire.

Features:
1. int8 quantization (5.48 tok/s)
2. End of Symbol for decode and speculative_decode (13.96 tok/s)
3. SQL context prefill cache (36.28 tok/s)

| Tag                     | tok/s | Mem      | Total time | Mean time |
|-------------------------|-------|----------|------------|-----------|
| 6B no-compile           | 5.48  | 11.44 GB | 1422.38sec | 4.23sec   |
| 6B+350M no-compile k=5  | 11.84 | 12.50 GB | 653.68sec  | 1.95sec   |
| 6B+350M no-compile k=20 | 13.96 | 12.51 GB | 571.17sec  | 1.70sec   |
| 6B+350M compile k=20    | 18.62 | 14.28 GB | 454.77sec  | 1.35sec   |
| 6B compile              | 19.65 | 11.44 GB | 407.97sec  | 1.21sec   |
| 6B compile_prefill      | 20.62 | 11.70 GB | 387.49sec  | 1.15sec   |
| 6B prefill cache        | 36.28 | 18.99 GB | 213.89sec  | 0.64sec   |

## Installation
[Download PyTorch nightly](https://pytorch.org/get-started/locally/)
Install sentencepiece and huggingface_hub
```bash
pip install sentencepiece huggingface_hub
```

To download nsql models, go to https://huggingface.co/NumbersStation/nsql-350M

## Benchmarks
Benchmarks run on a 2080ti, power limited to 300W.
Error is SQL levenshtein_distance of string.

| Tag                     | tok/s  | Mem      | Total time | Mean time | p95 time | Error | error<30  | Mean Accepted |
|-------------------------|--------|----------|------------|-----------|----------|-------|-----------|---------------|
| 350M no-compile         | 69.09  | 2.32 GB  | 107.71sec  | 0.32sec   | 0.63sec  | 38.10 | 0.54%     | -             |
| 2B no-compile           | 12.20  | 6.35 GB  | 593.51sec  | 1.77sec   | 3.35sec  | 34.65 | 0.59%     | -             |
| 6B no-compile           | 5.48   | 11.44 GB | 1422.38sec | 4.23sec   | 7.61sec  | 28.06 | 0.65%     | -             |
| 6B+350M no-compile k=5  | 11.84  | 12.50 GB | 653.68sec  | 1.95sec   | 3.17sec  | 28.06 | 0.65%     | 3.86          |
| 6B+350M no-compile k=20 | 13.96  | 12.51 GB | 571.17sec  | 1.70sec   | 3.15sec  | 28.06 | 0.65%     | 8.49          |
| 350M compile            | 209.47 | 2.33 GB  | 35.73sec   | 0.11sec   | 0.18sec  | 38.10 | 0.54%     | -             |
| 2B compile              | 38.97  | 6.35 GB  | 191.38sec  | 0.57sec   | 0.87sec  | 34.65 | 0.59%     | -             |
| 6B compile              | 19.65  | 11.44 GB | 407.97sec  | 1.21sec   | 1.82sec  | 28.06 | 0.65%     | -             |
| 6B+350M compile k=5     | 14.38  | 13.91 GB | 545.08sec  | 1.62sec   | 2.53sec  | 28.05 | 0.65%     | 3.87          |
| 6B+350M compile k=20    | 18.62  | 14.28 GB | 454.77sec  | 1.35sec   | 2.12sec  | 28.06 | 0.65%     | 8.48          |
| 350M compile_prefill    | 227.50 | 2.54 GB  | 32.77sec   | 0.10sec   | 0.17sec  | 38.10 | 0.54%     | -             |
| 6B compile_prefill      | 20.62  | 11.70 GB | 387.49sec  | 1.15sec   | 1.75sec  | 28.05 | 0.65%     | -             |
| 350M prefill cache      | 326.30 | 3.10 GB  | 22.47sec   | 0.07sec   | 0.12sec  | 38.18 | 0.54%     | -             |
| 6B prefill cache        | 36.28  | 18.99 GB | 213.89sec  | 0.64sec   | 1.10sec  | 28.06 | 0.65%     | -             |

## Convert model
```bash
python ./scripts/convert_hf_checkpoint.py --checkpoint_dir '<model-folder>/nsql-350M'
python ./quantize.py --checkpoint_dir '<model-folder>/nsql-350M' --mode int8
```

## Run spider test
```bash
python validation_spider.py --tag '350M no-compile' --checkpoint_path '<model-folder>/nsql-350M/model_int8.pth'
```

## GPTQ
todo

## License

see `gpt-fast`[BSD 3](https://github.com/pytorch-labs/gpt-fast/main/LICENSE) license.