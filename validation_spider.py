import contextlib
import itertools
import time
from pathlib import Path
import re
from typing import Optional

import torch
from datasets import load_dataset

from generate import _load_model, generate, encode_tokens
from prefill_cache import PrefillCache, PrefillCacheContext


def execute_validation(
        checkpoint_path,
        draft_checkpoint_path: Optional[Path] = None,
        is_compile=True,
        is_compile_prefill=True,
        use_tp=False,
        device='cuda',
        precision=torch.float32,
        max_seq_length=2000,
        speculate_k=5,
        prefill_cache=None,
        verbose=True,
        tag=""
):
    dataset = load_dataset('json', data_files="./spider_data/spider_nsql_dev.json")

    if verbose:
        print(dataset['train'][0])

    is_speculative = draft_checkpoint_path is not None

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp=False)
    if draft_checkpoint_path is not None:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path.parent)

    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
        'distance': [],
        'prefill_time': [],
        'draft_prefill_time': [],
        'time_use': [],
    }

    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])
    prof = contextlib.nullcontext()

    if is_compile:
        from generate import compile_this_file
        compile_this_file(is_speculative, use_tp, is_compile_prefill)


    def find_select_substring(s):

        pattern = r'SELECT.*;?'
        matches = re.findall(pattern, s, re.IGNORECASE)
        if len(matches) == 0:
            return s
        return matches[0]


    def normalize_sql(s):
        s = s[:-1] if s.endswith(";") else s
        return re.sub(" ,", ',', re.sub(" {2}", ' ', re.sub(" {2}", ' ', s.strip())))


    def levenshtein_distance(str1, str2):
        size_x = len(str1) + 1
        size_y = len(str2) + 1
        matrix = [[0]*size_y for _ in range(size_x)]
        for x in range(size_x):
            matrix[x][0] = x
        for y in range(size_y):
            matrix[0][y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if str1[x-1] == str2[y-1]:
                    matrix[x][y] = min(
                        matrix[x-1][y] + 1,
                        matrix[x-1][y-1],
                        matrix[x][y-1] + 1
                    )
                else:
                    matrix[x][y] = min(
                        matrix[x-1][y] + 1,
                        matrix[x-1][y-1] + 1,
                        matrix[x][y-1] + 1
                    )
        return matrix[-1][-1]


    count = 0
    for data in dataset['train']:
        question = data['prompt']
        context = data['context']
        label = data['response']
        prompt = f"{context}{question}\n\nSELECT"
        torch.cuda.synchronize()
        callback = lambda x: x
        t0 = time.perf_counter()

        encoded_context = encode_tokens(tokenizer, context, bos=False, device=device)
        encoded = encode_tokens(tokenizer, prompt, bos=False, device=device)
        prompt_length = encoded.size(0)
        encoded_context_length = encoded_context.size(0)

        if prefill_cache is not None:
            prefill_cache.set_context(PrefillCacheContext(context, encoded_context_length, model))

        with prof:
            y, metrics = generate(
                model,
                encoded,
                50,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=False,
                callback=callback,
                temperature=0.001,
                top_k=200,
                max_seq_length=max_seq_length,
                eos=tokenizer.eos_token_id,
                prefill_cache=prefill_cache,
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
            aggregate_metrics['prefill_time'].append(metrics['prefill_time'])
            aggregate_metrics['draft_prefill_time'].append(metrics['draft_prefill_time'])
        if count == 0:
            count += 1
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        t = time.perf_counter() - t0
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        aggregate_metrics['time_use'].append(t)
        result = tokenizer.decode(y.tolist())
        sql = find_select_substring(result)


        distance = levenshtein_distance(normalize_sql(sql.lower()), normalize_sql(label.lower()))
        aggregate_metrics['distance'].append(distance)

        count += 1
        if verbose:
            print(question)
            print(normalize_sql(sql))
            print(normalize_sql(label))
            print(f"Time for inference {count}/{dataset['train'].shape[0]}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec "
                  f"prompt_length:{prompt_length} tokens_generated:{tokens_generated}, distance: {distance},"
                  f" prefill_time: {aggregate_metrics['prefill_time'][-1]},  draft_prefill_time: {aggregate_metrics['draft_prefill_time'][-1]}, "
                  f" accept_counts: { aggregate_metrics['accept_counts'][-1]}, total : {sum(aggregate_metrics['accept_counts'][-1])}")
            print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

    print("="*20)
    if draft_checkpoint_path is not None:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
        if verbose:
            print(f"Acceptance probs: {acceptance_probs}")
            print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)]) / sum(counts_aggregated)}")
    else:
        counts_aggregated = [1]
        acceptance_probs = [1]

    import numpy as np
    dis = np.array(aggregate_metrics['distance'])
    time_use = np.array(aggregate_metrics['time_use'])
    if verbose:
        print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        print(f"Total time_use: {np.sum(time_use):.2f}")
        print(f"Average time_use: {np.mean(time_use):.2f}")
        print(f"Average distance: {np.mean(dis):.2f}")
        print(f"distance 0 rate: {np.count_nonzero(dis == 0) / dis.size:.02f}")
        print(f"distance < 5 rate: {np.count_nonzero(dis < 5) / dis.size:.02f}")
        print(f"distance < 10 rate: {np.count_nonzero(dis < 10) / dis.size:.02f}")
        print(f"distance < 30 rate: {np.count_nonzero(dis < 30) / dis.size:.02f}")

        print(f"Tag\tAverage tokens/sec:\tMemory used:\tTotal time_use:\tAverage time_use:\t"
              f"Average distance:\tdistance 0 rate:\tdistance < 5 rate:\t"
              f"distance < 10 rate:\tdistance < 30 rate:\tAcceptance probs\tMean Accepted")

    print(
          f"{tag}\t"
          f"{torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}\t"
          f"{torch.cuda.max_memory_reserved() / 1e9:.02f} GB\t"
          f"{np.sum(time_use):.2f}sec\t"
          f"{np.mean(time_use):.2f}sec\t"
          f"{np.percentile(time_use, 90):.2f}sec\t"
          f"{np.percentile(time_use, 95):.2f}sec\t"
          f"{np.percentile(time_use, 99):.2f}sec\t"
          f"{np.mean(dis):.2f}\t"
          f"{np.count_nonzero(dis == 0) / dis.size:.02f}%\t"
          f"{np.count_nonzero(dis < 5) / dis.size:.02f}%\t"
          f"{np.count_nonzero(dis < 10) / dis.size:.02f}%\t"
          f"{np.count_nonzero(dis < 30) / dis.size:.02f}%\t"
          # f"{[round(num, 2) for num in acceptance_probs]}\t"
          f"{sum([idx * i for idx, i in enumerate(counts_aggregated)]) / sum(counts_aggregated):.2f}\t")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Your CLI description.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path('/home/vendala/_Models/nsql-350M/model_int8.pth'), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--prefill_cache', action='store_true', help='Whether to use prefill cache')
    parser.add_argument('--max_seq_length', type=int, default=2000, help='Max seq length.')
    parser.add_argument('--verbose', action='store_true', help='show every result')
    parser.add_argument('--tag', type=str, default="default run", help='use for log tag')

    args = parser.parse_args()

    execute_validation(
        checkpoint_path=args.checkpoint_path,
        draft_checkpoint_path=args.draft_checkpoint_path,
        is_compile=args.compile,
        is_compile_prefill=args.compile_prefill,
        max_seq_length=args.max_seq_length,
        speculate_k=args.speculate_k,
        prefill_cache=PrefillCache("cuda:0", "cuda:0", 2) if args.prefill_cache else None,
        verbose=args.verbose,
        tag=args.tag
    )

