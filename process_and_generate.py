import pandas as pd
import torch
from evodiff.pretrained import OA_DM_38M, OA_DM_640M
from evodiff.conditional_generation import inpaint_multiple_regions
from collections import defaultdict
import os
import shutil
import argparse
import session_info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_activesite_data(path):
    df = pd.read_excel(path)
    df.dropna(subset=['Active site'], inplace=True)
    df['Active site'] = df['Active site'].apply(lambda x: [int(i[9:]) for i in x.split('; ') if i.startswith('ACT_SITE')])
    return df

def mask_sequences(sequences, start_ids, end_ids):
        masked_sequences = []
        for sequence, starts, ends in zip(sequences, start_ids, end_ids):
            masked_sequence = sequence
            offset = 0
            for start, end in zip(starts, ends):
                start += offset
                end += offset
                masked_sequence = masked_sequence[:start] + '#' * (end - start) + masked_sequence[end:]
                offset += (end - start) - (end - start)
            masked_sequences.append(masked_sequence)
        return masked_sequences

def tokenize_sequences(sequences, tokenizer, device=device):
    tokenized_sequences = [torch.tensor(tokenizer.tokenizeMSA(seq)) for seq in sequences]
    tokenized_sequences = [seq.to(device) for seq in tokenized_sequences]
    return tokenized_sequences

def prepare_indices(start_ids, end_ids, device=device):
    start_idxs = torch.tensor(start_ids).to(device)
    end_idxs = torch.tensor(end_ids).to(device)
    return start_idxs, end_idxs

def generate_sequences(df, K, model_name):
    if model_name == 'OADM_640M':
        checkpoint = OA_DM_640M()
    else:
        checkpoint = OA_DM_38M()
    model, collater, tokenizer, scheme = checkpoint
    model.to(device)

    activesite_idxs = [[(x-1) for x in row] for row in df['Active site'].values.tolist()]
    sequences = df['Sequence'].values.tolist()
    start_ids = activesite_idxs
    end_ids = df['Active site'].values.tolist()

    # Mask the sequences
    masked_sequences = mask_sequences(sequences, start_ids, end_ids)

    # Tokenize the masked sequences
    tokenized_sequences = tokenize_sequences(masked_sequences, tokenizer, device)

    sequences = [ele for ele in sequences for i in range(K)]
    tokenized_sequences = [ele for ele in tokenized_sequences for i in range(K)]
    start_ids = [ele for ele in start_ids for i in range(K)]
    end_ids = [ele for ele in end_ids for i in range(K)]

    untokenized_seqs, sequences, untokenized_idrs, sequences_idrs, save_starts, save_ends = inpaint_multiple_regions(model, tokenized_sequences, start_ids, end_ids, sequences, tokenizer)

    generated_map = defaultdict((lambda : []))
    
    for i, sequence in enumerate(sequences):
        sequence = sequence[0]
        generated_map[sequence].append(untokenized_seqs[i][0])

    return generated_map
    
def main(activesite_path, n_sample_query, K_generate, model_name):
    df = load_activesite_data(activesite_path)

    df_sub = df.sample(n=n_sample_query, random_state=42)
    generated_map = generate_sequences(df_sub, K_generate, model_name)
    query_seqs = generated_map.keys()

    if os.path.exists('generated_seqs'):
        shutil.rmtree('generated_seqs')
    os.makedirs('generated_seqs')

    for i, query_seq in enumerate(query_seqs):
        generated_seqs = generated_map[query_seq]
        with open(f'generated_seqs/SEQUENCE_{i}.fasta', "w") as f:
            f.write(">" + f'QUERY_SEQUENCE_{i}' + "\n" + query_seq + "\n\n")
            for j, seq in enumerate(generated_seqs):
                f.write(">" + f'GENERATED_SEQUENCE_{i}_{j}' + "\n" + seq + "\n\n")

    print('Saved the Generated Sequences in generated_seqs as fasta files!')

if __name__== "__main__":
    parser = argparse.ArgumentParser(description ='Generate Protein Sequences using EvoDiff and Active Site Data')
    parser.add_argument('--path', 
                        help ='Enter the path to the active site data',
                        type=str,
                        required=True)
    parser.add_argument('-n --n_samples', dest ='n_sample_query', 
                        type=int,
                        required=True,
                        help ='Enter the number of original sequences to sample')
    parser.add_argument('-k', dest ='K_generate', 
                        type=int,
                        default=5,
                        help ='Enter the number of protein sequences to generate per original sample')
    parser.add_argument('--model -m', dest ='model_name',
                        choices = ['OADM_640M', 'OADM_38M'],
                        default= 'OADM_38M',
                        help ='Enter the number of protein sequences to generate per original sample')
         
    args = parser.parse_args()
    main(args.path, args.n_sample_query, args.K_generate, args.model_name)
    session_info.show(write_req_file=True, req_file_name='requirements.txt')
