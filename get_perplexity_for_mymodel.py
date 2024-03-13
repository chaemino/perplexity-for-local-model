import re
import sys
import json
import torch

import argparse

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import transformers 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluate():
    def __init__(self, args):

        self.model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path)
        self.model = self.model.to(device)
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
        if 'gpt' in args.model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_length = args.max_length
#         self.max_length = self.tokenizer.model_max_length
        self.batch_size = args.batch_size

        self.data_dir = args.data_dir_path
        self.test_data = args.test_data_file
        self.data = self.dataload(self.data_dir, self.test_data)


    def perplexity(self):

        ### dataload
        data = self.data

        encodings = self.tokenizer(
                data,
                return_tensors="pt", 
                padding="max_length", 
                max_length=self.max_length, 
                truncation=True,
                return_attention_mask=True
                ).to(device)
        input_ids = encodings.input_ids
        attn_masks = encodings.attention_mask

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in (range(0, len(input_ids), self.batch_size)):
            end_index = min(start_index+self.batch_size, len(input_ids))
            encoded_batch = input_ids[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]
            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits
            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                    (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                    / shift_attention_mask_batch.sum(1)
                    )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

    def dataload(self, data_dir=None, test_data=None): ### need custom for each task
        ## load data
        with open(data_dir+test_data, 'r') as f:
            data = f.readlines()
        data = [sent.replace('\n', '').replace('\"', '') for sent in data]
        return data

    def clean_string(self, text): ## need custom for each task
        text = re.sub('[ \t\r\n]+', ' ', text)
        text = text.replace('\uf85e','')
        text = re.sub('  *',' ',text).strip() ## 다중 공백 제거
        text = text.strip()
        return text


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Sem Sim')
    parser.add_argument('--model_path', type=str, required=False, help='generation model path')
    parser.add_argument('--prompt_model_path', type=str, required=False, help='prompt model path')
    parser.add_argument('--batch_size', '-bs', type=int, required=True, help='batch_size')
    parser.add_argument('--max_length', '-ml', type=int, required=True, help='max length')
    parser.add_argument('--data_dir_path', type=str, required=False, help='prediction file data dictionary')
    parser.add_argument('--test_data_file', type=str, required=False, help='prediction file name ex){filename}.jsonl')

    args = parser.parse_args()

    evaluate = Evaluate(args)
#     perplexity = evaluate.perplexity_torchmetrics()
    perplexity = evaluate.perplexity()
    print(round(perplexity["mean_perplexity"], 2))
    print(perplexity["perplexities"][:5])


