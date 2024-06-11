import pandas as pd
from evodiff.pretrained import OA_DM_38M, OA_DM_640M
import torch
from evodiff.conditional_generation import inpaint_multiple_regions
from collections import defaultdict
import pickle as pkl

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

def generate_sequences(df, model='OADM_38M'):
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

    sequences = [ele for ele in sequences for i in range(5)]
    tokenized_sequences = [ele for ele in tokenized_sequences for i in range(5)]
    start_ids = [ele for ele in start_ids for i in range(5)]
    end_ids = [ele for ele in end_ids for i in range(5)]

    untokenized_seqs, sequences, untokenized_idrs, sequences_idrs, save_starts, save_ends = inpaint_multiple_regions(model, tokenized_sequences, start_ids, end_ids, sequences, tokenizer)

    generated_map = defaultdict((lambda : []))
    
    for i, sequence in enumerate(sequences):
        sequence = sequence[0]
        generated_map[sequence].append(untokenized_seqs[i][0])

    return generated_map
    
def main():
    activesite_path = '/content/uniprotkb_active_site_AND_reviewed_true_2024_05_31.xlsx'
    df = load_activesite_data(activesite_path)

    df_sub = df.sample(n=100, random_state=42)
    generated_map = generate_sequences(df_sub)
    
    with open('generated_sequences.pkl', 'wb') as f:
        pkl.dump(dict(generated_map), f)

    print('Saved the Generated Sequences as generated_sequences.pkl!')

if __name__== "__main__":
    main()
