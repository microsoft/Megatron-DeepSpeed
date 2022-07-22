import zstandard
import sys
import time
import os

def pile_download(download_url, file_path, i):
    start = time.time()
    zstd_file_path = f"{file_path}{i:02}.jsonl.zst"
    download_path = f"{download_url}{i:02}.jsonl.zst"
    if not os.path.exists(zstd_file_path):
        os.system(f"wget -P {file_path} {download_path}")
        print("Finished downloading chunk {} in {} sec".format(
            i, time.time() - start))

def pile_decompress(download_url, file_path, i):
    zstd_file_path = f"{file_path}{i:02}.jsonl.zst"
    output_path = f"{file_path}{i:02}.jsonl"
    if not os.path.exists(output_path):
        if not os.path.exists(zstd_file_path):
            pile_download(download_url, file_path, i)
        start = time.time()
        with open(zstd_file_path, 'rb') as compressed:
            decomp = zstandard.ZstdDecompressor()
            with open(output_path, 'wb') as destination:
                decomp.copy_stream(compressed, destination)
        os.remove(zstd_file_path)
        print("Finished decompressing chunk {} in {} sec".format(
            i, time.time() - start))

def pile_preprocess(download_url, file_path, vocab_file, num_workers, i,
    num_retry=3):
    json_file_path = f"{file_path}{i:02}.jsonl"
    output_prefix = f"{file_path}pile_bert_train_{i:02}"
    if not os.path.exists(f"{output_prefix}_text_sentence.idx"):
        if not os.path.exists(json_file_path):
            pile_decompress(download_url, file_path, i)
        start = time.time()
        cmd = f"python ../../tools/preprocess_data.py \
                --input {json_file_path} \
                --output-prefix {output_prefix} \
                --vocab {vocab_file} \
                --dataset-impl mmap \
                --tokenizer-type BertWordPieceLowerCase \
                --split-sentences \
                --workers {num_workers} "
        # Somehow it's possible to hit MemoryError during above cmd even if
        # there is still memory available. Currently just delete the bad output
        # and retry a few times. TODO: improve this.
        for t in range(num_retry):
            if os.system(cmd) == 0: # Success
                os.remove(json_file_path)
                break
            else:
                print(f"Error: chunk {i} preprocessing got error, delete bad output and retry {t}")
                if os.path.exists(f"{output_prefix}_text_sentence.idx"):
                    os.remove(f"{output_prefix}_text_sentence.idx")
                if os.path.exists(f"{output_prefix}_text_sentence.bin"):
                    os.remove(f"{output_prefix}_text_sentence.bin")
        if not os.path.exists(f"{output_prefix}_text_sentence.idx"):
            print(f"Error: chunk {i} preprocessing got error, tried {t} times but still failed")
        print("Finished preprocessing chunk {} in {} sec".format(
            i, time.time() - start))

if __name__ == '__main__':
    # The raw Pile data has 30 compressed .zst chunks. To run on single
    # machine for all chunks, run "python prepare_pile_data.py range 0 30".
    # You can also split and run on multiple machines to speed up, since
    # processing one chunk can take hours. The whole process only uses CPU.
    num_chunk = 30
    if sys.argv[1] == "range":
        # "python prepare_pile_data.py range 0 30" means process chunk 0 to 29.
        selected_chunk = range(int(sys.argv[2]), int(sys.argv[3]))
    else:
        # "python prepare_pile_data.py 2 5 8" means process chunk 2, 5 and 8.
        selected_chunk = [int(x) for x in sys.argv[1:]]
    print("selected_chunk: ", selected_chunk)
    # Number of process when preprocessing. Adjust based on your CPU.
    num_workers = 40
    # Where the raw Pile data can be downloaded. The url may change in future.
    # Contact EleutherAI (https://github.com/EleutherAI/the-pile) if you cannot
    # fine the data in this url.
    download_url = "https://the-eye.eu/public/AI/pile/train/"
    # Path to download and store all the output files during the whole process.
    # Estimated max space usage would be around 1.6 TB. Estimated max memory
    # usage would be O(100GB).
    file_path = "/vc_data_blob/users/conglli/the_pile_bert/"
    vocab_file = "bert-large-uncased-vocab.txt"
    vocab_url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt"
    if not os.path.exists(vocab_file):
        os.system(f"wget {vocab_url}")
    os.makedirs(file_path, exist_ok=True)

    for i in selected_chunk:
        pile_preprocess(download_url, file_path, vocab_file, num_workers, i)
