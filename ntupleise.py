#!/usr/bin/env python
"""
"""
import argparse
from typing import Final
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
import uproot.writing
import h5py as h5
import tqdm


ELECTRON_PID: Final[int] = 11
MUON_PID: Final[int] = 13

BRANCH_LIST: Final[list[str]] = [
    # EFlowTrack
    "EFlowTrack/EFlowTrack.PT",
    "EFlowTrack/EFlowTrack.Eta",
    "EFlowTrack/EFlowTrack.Phi",
    "EFlowTrack/EFlowTrack.PID",
    "EFlowTrack/EFlowTrack.Charge",
    "EFlowTrack/EFlowTrack.IsRecoPU", # TODO
    # EFlowPhoton
    "EFlowPhoton/EFlowPhoton.ET",
    "EFlowPhoton/EFlowPhoton.Eta",
    "EFlowPhoton/EFlowPhoton.Phi",
    # EFlowNeutralHadron
    "EFlowNeutralHadron/EFlowNeutralHadron.ET",
    "EFlowNeutralHadron/EFlowNeutralHadron.Eta",
    "EFlowNeutralHadron/EFlowNeutralHadron.Phi",
    # MET
    "MissingET/MissingET.MET",
    "MissingET/MissingET.Phi",
    # PUPPI MET
    "PuppiMissingET/PuppiMissingET.MET",
    "PuppiMissingET/PuppiMissingET.Phi",
    # Generated MET
    "GenMissingET/GenMissingET.MET",
    "GenMissingET/GenMissingET.Phi",
    #
    "GenPileUpMissingET/GenPileUpMissingET.MET",
    "GenPileUpMissingET/GenPileUpMissingET.Phi",
    #
    'Weight/Weight.Weight',
]

PREFIX_ALIAS_DICT: Final[dict[str, str]] = {
    'EFlowTrack': 'track',
    'EFlowPhoton': 'photon',
    'EFlowNeutralHadron': 'neutral_hadron',
    'MissingET': 'pf_met',
    'PuppiMissingET': 'puppi_met',
    'GenMissingET': 'gen_met',
    'GenPileUpMissingET': 'gen_pileup_met',
    'Weight': 'gen',
}

FEATURE_ALIAS_DICT: Final[dict[str, str]] = {
    'ET': 'PT',
    'MET': 'PT',
    'IsRecoPU': 'is_reco_pu',
}


def make_alias(
    branch: str
) -> str:
    """
    """
    prefix, feature = branch.split('/')[1].split('.')
    prefix = PREFIX_ALIAS_DICT[prefix]
    feature = FEATURE_ALIAS_DICT.get(feature, feature)
    feature = feature.lower()
    alias = f'{prefix}_{feature}'
    return alias


def run(
    input_path_list: list[Path],
    output_path: Path,
    input_treepath: str = 'Delphes',
    input_branch_list: list[str] = BRANCH_LIST,
) -> None:
    """
    """
    aliases: dict[str, str] = {make_alias(each): each
                               for each in input_branch_list}
    expressions: list[str] = list(aliases.keys())

    input_files = {each: input_treepath for each in input_path_list}
    total = sum(num_entries for *_, num_entries in uproot.num_entries(input_files))

    output_file = h5.File(output_path, 'w')

    def create_dataset(name, dtype: type = np.float32, vlen: bool = False):
        """create variable-length dataset"""
        if vlen:
            dtype = h5.vlen_dtype(dtype)
        return output_file.create_dataset(name=name, shape=total, dtype=dtype)

    output = {}

    track_suffix_list = ['px', 'py', 'eta', 'is_electron', 'is_muon',
                         'is_hadron', 'charge', 'is_reco_pu']
    for suffix in track_suffix_list:
        key = f'track_{suffix}'
        dtype = np.float32 if suffix in ['px', 'py', 'eta'] else np.int64
        output[key] = create_dataset(key, dtype, vlen=True)

    tower_suffix_list = ['px', 'py', 'eta', 'is_hadron']
    for suffix in tower_suffix_list:
        key = f'tower_{suffix}'
        dtype = np.float32 if suffix in ['px', 'py', 'eta'] else np.int64
        output[key] = create_dataset(key, dtype, vlen=True)

    for obj in ['pf_met', 'puppi_met', 'gen_met', 'gen_pileup_met']:
        for feature in ['pt', 'phi']:
            key = f'{obj}_{feature}'
            output[key] = create_dataset(key)

    for key in ['gen_weight']:
        output[key] = create_dataset(key)

    start = 0
    stop = 0

    with tqdm.tqdm(total=total) as progress_bar:
        chunk_iterator = uproot.iterate(
            files=input_files,
            expressions=expressions,
            aliases=aliases
        )
        for chunk in chunk_iterator:
            num_entries = len(chunk)
            start = stop
            stop = start + num_entries
            slicing = slice(start, stop)

            output['track_px'][slicing] = chunk['track_pt'] * np.cos(chunk['track_phi'])
            output['track_py'][slicing] = chunk['track_pt'] * np.sin(chunk['track_phi'])

            for suffix in ['eta', 'is_reco_pu', 'charge']:
                key = f'track_{suffix}'
                output[key][slicing] = chunk[key]

            track_abs_pid = np.abs(chunk['track_pid'])
            output['track_is_electron'][slicing] = track_abs_pid == ELECTRON_PID
            output['track_is_muon'][slicing] = track_abs_pid == MUON_PID
            output['track_is_hadron'][slicing] = (track_abs_pid != ELECTRON_PID) & (track_abs_pid != MUON_PID)

            for feature in ['pt', 'eta', 'phi']:
                arrays = [ak.values_astype(chunk[f'{prefix}_{feature}'], np.float32)
                          for prefix in ['neutral_hadron', 'photon']]
                chunk[f'tower_{feature}'] = ak.concatenate(
                    arrays=arrays,
                    axis=1
                )

            output['tower_px'][slicing] = chunk['tower_pt'] * np.cos(chunk['tower_phi'])
            output['tower_py'][slicing] = chunk['tower_pt'] * np.sin(chunk['tower_phi'])

            output['tower_eta'][slicing] = chunk['tower_eta']
            output['tower_is_hadron'][slicing] = ak.concatenate(
                arrays=[
                    ak.ones_like(chunk['neutral_hadron_pt'], dtype=np.int64),
                    ak.zeros_like(chunk['photon_pt'], dtype=np.int64)
                ],
                axis=1,
            )

            for obj in ['pf_met', 'puppi_met', 'gen_met', 'gen_pileup_met']:
                for feature in ['pt', 'phi']:
                    key = f'{obj}_{feature}'
                    output[key][slicing] = ak.values_astype(
                        array=ak.flatten(chunk[key]),
                        to=np.float32
                    )

            for key in ['gen_weight']:
                array = ak.to_numpy(chunk[key][:, 0])
                output[key][slicing] = ak.values_astype(
                    array=array,
                    to=np.float32,
                )

            progress_bar.update(n=num_entries)

    output_file.close()

def run_batch(
    dataset: str,
    counter: int,
    delphes_config: str,
) -> None:
    """
    """
    root = Path('/store/hep/users/slowmoyang/KSC-RnD-2024/')

    counter_label = f'{counter:06d}'

    input_dir = root / 'delphes' / delphes_config / dataset / counter_label
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    input_path_list = sorted(
        input_dir.glob('*.root'),
        key=lambda item: int(item.stem.removeprefix('output_'))
    )

    output_dir = root / 'ntuple' / delphes_config / dataset
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_path = output_dir / counter_label
    output_path = output_path.with_suffix('.h5')

    run(
        input_path_list=input_path_list,
        output_path=output_path
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers()

    batch_parser = subparsers.add_parser('batch')
    batch_parser.add_argument('-d', '--dataset', type=str, required=True,
                              help='dataset')
    batch_parser.add_argument('-c', '--counter', type=int, required=True,
                              help='counter')
    batch_parser.add_argument('--delphes-config', type=str,
                              default='CMS_PhaseII_200PU_v04',
                              help='delphes config')
    batch_parser.set_defaults(func=run_batch)

    args = parser.parse_args()
    args = vars(args)

    func = args.pop('func')
    func(**args)


if __name__ == '__main__':
    args = main()
