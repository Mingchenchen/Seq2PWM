import os
from importlib.util import find_spec
from typing import Any, Dict, Union, List

import torch
import torch.nn as nn


ESM_OUTPUT_DIM = {
        "esm1b_t33_650M_UR50S": 1280,
        "esm1v_t33_650M_UR90S_1": 1280,
        "esm2_t6_8M_UR50D": 320,
        "esm2_t12_35M_UR50D": 480,
        "esm2_t30_150M_UR50D": 640,
        "esm2_t33_650M_UR50D": 1280,
        "esm2_t36_3B_UR50D": 2560,
        "esm2_t48_15B_UR50D": 5120,
    }
ACIDS = 'XACDEFGHIKLMNPQRSTVWY'
ressymb_to_resindex = {acid: index for index, acid in enumerate(ACIDS)}
# index_to_acids = {index: acid for acid, index in acids_to_index.items()}
# ressymb_to_resindex = {
#     'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
#     'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
#     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
#     'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
#     'X': 20,
#     }
resindex_to_ressymb = {}
for k, v in ressymb_to_resindex.items():
    resindex_to_ressymb[v] = k


def flatten(lst: List[List[Any]]) -> List[Any]:
    return [item for sublist in lst for item in sublist]



def compute_md5(file_path, chunk_size=65536) -> str:
    import hashlib

    md5 = hashlib.md5()
    with open(file_path, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = fin.read(chunk_size)
    return md5.hexdigest()


def download(url, path, save_file=None, md5=None):
    """Return a verified local file, downloading it only when it is absent.

    Parameters:
        url (str): URL to download.
        path (str): Directory in which the file is stored.
        save_file (str, optional): Output filename; inferred from the URL by default.
        md5 (str, optional): Expected MD5 checksum.
    """
    from urllib.request import urlretrieve

    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    if save_file is None:
        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[: save_file.find("?")]
    save_file = os.path.join(path, save_file)

    if os.path.exists(save_file):
        actual_md5 = compute_md5(save_file)
        if md5 is None or actual_md5 == md5:
            print(f"Using local ESM weights: {save_file}")
            return save_file
        raise RuntimeError(
            f"Existing ESM weight file has an unexpected MD5 checksum: {save_file}. "
            f"Expected {md5}, found {actual_md5}. Remove or replace this file."
        )

    partial_file = f"{save_file}.part"
    print(f"Downloading {url} to {save_file}")
    try:
        urlretrieve(url, partial_file)
        if md5 is not None:
            actual_md5 = compute_md5(partial_file)
            if actual_md5 != md5:
                raise RuntimeError(
                    f"Downloaded ESM weights failed MD5 verification. "
                    f"Expected {md5}, found {actual_md5}."
                )
        os.replace(partial_file, save_file)
    finally:
        if os.path.exists(partial_file):
            os.remove(partial_file)
    return save_file




class EvolutionaryScaleModeling(nn.Module):
    """The protein language model, Evolutionary Scale Modeling (ESM) proposed in `Biological
    Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein
    Sequences`_.

    .. _Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences:
        https://www.biorxiv.org/content/10.1101/622803v1.full.pdf

    .. _Language models of protein sequences at the scale of evolution enable accurate structure prediction:
        https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf
    """

    url = {
        "esm1b_t33_650M_UR50S": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt",
        "esm1v_t33_650M_UR90S_1": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt",
        "esm1b_t33_650M_UR50S-contact-regression": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt",
        "esm2_t6_8M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
        "esm2_t12_35M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
        "esm2_t30_150M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
        "esm2_t33_650M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "ESM-2-3B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
        "ESM-2-15B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt",
    }

    md5 = {
        "esm1b_t33_650M_UR50S": "ba8914bc3358cae2254ebc8874ee67f6",
        "esm1v_t33_650M_UR90S_1": "1f04c2d2636b02b544ecb5fbbef8fefd",
        "esm1b_t33_650M_UR50S-contact-regression": "e7fe626dfd516fb6824bd1d30192bdb1",
        "esm2_t6_8M_UR50D": "8039fc9cee7f71cd2633b13b5a38ff50",
        "esm2_t12_35M_UR50D": "a894ddb31522e511e1273abb23b5f974",
        "esm2_t30_150M_UR50D": "229fcf8f9f3d4d442215662ca001b906",
        "esm2_t33_650M_UR50D": "ba6d997e29db07a2ad9dca20e024b102",
        "esm2_t36_3B_UR50D": "d37a0d0dbe7431e48a72072b9180b16b",
        "esm2_t48_15B_UR50D": "af61a9c0b792ae50e244cde443b7f4ac",
    }

    output_dim = {
        "esm1b_t33_650M_UR50S": 1280,
        "esm1v_t33_650M_UR90S_1": 1280,
        "esm2_t6_8M_UR50D": 320,
        "esm2_t12_35M_UR50D": 480,
        "esm2_t30_150M_UR50D": 640,
        "esm2_t33_650M_UR50D": 1280,
        "esm2_t36_3B_UR50D": 2560,
        "esm2_t48_15B_UR50D": 5120,
    }

    num_layer = {
        "esm1b_t33_650M_UR50S": 33,
        "esm1v_t33_650M_UR90S_1": 33,
        "esm2_t6_8M_UR50D": 6,
        "esm2_t12_35M_UR50D": 12,
        "esm2_t30_150M_UR50D": 30,
        "esm2_t33_650M_UR50D": 33,
        "esm2_t36_3B_UR50D": 36,
        "esm2_t48_15B_UR50D": 48,
    }

    def __init__(self, path, model="esm2_t33_650M_UR50D", freeze_n_layers: int = 0):
        """Freeze parameters except transformer layers above ``freeze_n_layers``."""
        super().__init__()
        if find_spec("esm") is None:
            raise ImportError("To use ESM model you must install fair-esm first")

        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)
        self.path = path

        _model, alphabet = self.load_weight(path, model)
        self.alphabet = alphabet
        self.output_dim = self.output_dim[model]
        self.model = _model
        self.repr_layer = self.num_layer[model]
        if freeze_n_layers == -1:
            freeze_n_layers = len(list(self.model.named_parameters())) + 1 #539
        print(f"[INFO] the freeze_n_layers is {freeze_n_layers}")
        for name, param in self.model.named_parameters():
            base_name, *other = name.split(".")
            if base_name == "layers" and (int(other[0]) + 1 > freeze_n_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.batch_converter = alphabet.get_batch_converter()

    def load_weight(self, path, model):
        if model not in self.url:
            raise ValueError("Unknown model `%s`" % model)
        model_file = download(self.url[model], path, md5=self.md5[model])
        model_data = torch.load(model_file, map_location='cpu')
        if model != "esm1v_t33_650M_UR90S_1" and not model.startswith("esm2"):
            regression_model = "%s-regression" % model
            regression_file = download(
                self.url[regression_model], path, md5=self.md5[regression_model]
            )
            regression_data = torch.load(regression_file, map_location="cpu")
        else:
            regression_data = None
        model_name = os.path.basename(self.url[model])

        import esm

        return esm.pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data)

    def convert_to_esm_input(self, sequences, device):
        # Convert tensors to sequence strings.
        # seq_list = []
        # for seq in sequences:
        #     seq_str = ''.join([resindex_to_ressymb[int(i.item())] for i in seq])
        #     seq_list.append(seq_str)
        
        # seq_input = [(str(idx), seq) for idx, seq in enumerate(seq_list)]
        # _, _, batch_tokens = self.batch_converter(seq_input)
        # batch_tokens = batch_tokens.to(device)
        # return batch_tokens
        seq_input = [(str(idx), seq) for idx, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(seq_input)
        batch_tokens = batch_tokens.to(device)
        return batch_tokens


class ESMAsSchNetEmbedding(EvolutionaryScaleModeling):
    def __init__(self, path=".cache/esm", model="esm2_t33_650M_UR50D", freeze_n_layers=-1):
        super().__init__(path, model, freeze_n_layers)

    def forward(self, batch_aa):
        
        # if isinstance(batch, Batch):
        #     residues_lists = [seq.split("|") for seq in batch.residue_seq_str]
        #     interface_masks = [seq.split("|") for seq in batch.interface_mask_str]
        # elif isinstance(batch, dict):
        #     residues_lists = [seq.split("|") for seq in batch["chain_residues_list"]]
        #     interface_masks = [seq.split("|") for seq in batch["interface_masks"]]
        # else:
        #     raise ValueError(f"unsupported batch input type {type(batch)}")
        seq_list = []
        for batch in batch_aa:
            # import pdb
            # pdb.set_trace()
            # seq_list.append(batch)
            # try:
            seq_list.append(''.join([resindex_to_ressymb.get(int(aa),'X') for aa in batch]))
            # except:
            #     print(batch)
            #     seq_list.append(''.join([resindex_to_ressymb.get(int(aa),'') for aa in batch]))
                # import pdb
                # pdb.set_trace()
        # import pdb;pdb.set_trace()
        residues_lists = [[seq] for seq in seq_list]
        seq_flatten = flatten(residues_lists)
        batch_tokens = self.convert_to_esm_input(seq_flatten, device=batch_aa.device)
        results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
        representations: torch.Tensor = results["representations"][self.repr_layer]
        if self.alphabet.prepend_bos and self.alphabet.append_eos:
            representations = representations[:, 1:-1, :]
        elif self.alphabet.prepend_bos:
            representations = representations[:, 1:, :]
        elif self.alphabet.append_eos:
            representations = representations[:, :-1, :]

        # interface_masks = torch.tensor(
        #     [
        #         [bool(int(mask)) for mask in mask_seq]
        #         + [False] * (representations.shape[1] - len(mask_seq))
        #         for mask_seq in flatten(interface_masks)
        #     ],
        #     device=batch.y.device,
        # )
        # masked_representations = representations.masked_select(
        #     interface_masks.unsqueeze(2)
        # ).reshape(-1, self.output_dim)

        return representations  # (L, F)

if __name__ == '__main__':
    import torch
    model = ESMAsSchNetEmbedding()
    # import pdb
    # pdb.set_trace()
    s = torch.tensor([[ 8,  9,  2, 2,  4, 17, 2,  4, 11,  9,  9, 11,10, 17,  7,  12, 16,  9]]).cuda()
    print(s.shape)
    output = model(s)
    print(output.shape)

