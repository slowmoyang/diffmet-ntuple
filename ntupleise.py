#!/usr/bin/env python
"""
"""
import argparse
from enum import Enum
from typing import Final
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
import uproot.writing
import h5py as h5
import tqdm


# FIXME:
PROJECT_DATA_DIR = Path('/store/hep/users/slowmoyang/KSC-RnD-2024/')


BRANCH_LIST: Final[list[str]] = [
    # EFlowTrack
    "EFlowTrack/EFlowTrack.PT",
    "EFlowTrack/EFlowTrack.Eta",
    "EFlowTrack/EFlowTrack.Phi",
    "EFlowTrack/EFlowTrack.PID",
    "EFlowTrack/EFlowTrack.Charge",
    "EFlowTrack/EFlowTrack.IsRecoPU", # TODO
    "EFlowTrack/EFlowTrack.D0",
    "EFlowTrack/EFlowTrack.DZ",
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
    'PID': 'pdgid',
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


class ParticleTypeIndex(Enum):
    ZERO_PAD = 0
    ELECTRON = 1
    MUON = 2
    CHARGED_HADRON = 3
    PHOTON = 4
    NEUTRAL_HADRON = 5


class PDGId(Enum):
    ELECTRON = 11
    MUON = 13



def _convert_track_pdgid_to_pid(pdgid_arr):
    pdgid_arr = ak.to_numpy(np.abs(pdgid_arr))

    electron_mask = pdgid_arr == PDGId.ELECTRON.value
    muon_mask = pdgid_arr == PDGId.MUON.value

    pid_arr = np.full_like(
        pdgid_arr,
        fill_value=ParticleTypeIndex.CHARGED_HADRON.value,
    )

    pid_arr[electron_mask] = PDGId.ELECTRON.value
    pid_arr[muon_mask] = PDGId.MUON.value

    return pid_arr

def convert_track_pdgid_to_pid(pdgid_arr):
    return ak.Array(map(_convert_track_pdgid_to_pid, pdgid_arr))


def run(
    input_file_path_list: list[Path],
    output_file_path: Path,
    input_treepath: str = 'Delphes',
    input_branch_list: list[str] = BRANCH_LIST,
) -> None:
    """
    """
    aliases: dict[str, str] = {make_alias(each): each
                               for each in input_branch_list}
    expressions: list[str] = list(aliases.keys())

    input_files = {each: input_treepath for each in input_file_path_list}
    total = sum(num_entries for *_, num_entries in uproot.num_entries(input_files))

    output_file = h5.File(output_file_path, 'w')

    def create_dataset(name, dtype: type = np.float32, vlen: bool = False):
        """create variable-length dataset"""
        if vlen:
            dtype = h5.vlen_dtype(dtype)
        return output_file.create_dataset(name=name, shape=total, dtype=dtype)

    output = {}

    ###########################################################################
    # createt datasets for tracks
    ###########################################################################
    track_float32_suffix_list = [
        'px', 'py', 'eta', 'd0', 'dz'
    ]

    track_int64_suffix_list = [
        'is_reco_pu', 'charge', 'pid',
    ]
    track_suffix_list = track_float32_suffix_list + track_int64_suffix_list
    for suffix in track_suffix_list:
        key = f'track_{suffix}'
        dtype = np.float32 if suffix in track_float32_suffix_list else np.int64
        output[key] = create_dataset(key, dtype, vlen=True)

    ###########################################################################
    # create datasets for towers
    ###########################################################################
    tower_float32_suffix_list = ['px', 'py', 'eta']
    tower_int64_suffix_list = ['pid']
    tower_suffix_list = tower_float32_suffix_list + tower_int64_suffix_list

    for suffix in tower_suffix_list:
        key = f'tower_{suffix}'
        dtype = np.float32 if suffix in tower_float32_suffix_list else np.int64
        output[key] = create_dataset(key, dtype, vlen=True)

    ###########################################################################
    # create datasets for various types of METs
    ###########################################################################
    for obj in ['pf_met', 'puppi_met', 'gen_met', 'gen_pileup_met']:
        for feature in ['pt', 'phi']:
            key = f'{obj}_{feature}'
            output[key] = create_dataset(key)

    ###########################################################################
    #
    ###########################################################################
    for key in ['gen_weight']:
        output[key] = create_dataset(key)

    ###########################################################################
    # fill datasets
    ###########################################################################
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

            ####################################################################
            # track
            ####################################################################
            track_pt = chunk['track_pt'] # type: ignore
            track_phi = chunk['track_phi'] # type: ignore
            output['track_px'][slicing] = track_pt * np.cos(track_phi)
            output['track_py'][slicing] = track_pt * np.sin(track_phi)

            for suffix in ['eta', 'd0', 'dz', 'is_reco_pu', 'charge']:
                key = f'track_{suffix}'
                output[key][slicing] = chunk[key] # type: ignore

            output['track_pid'][slicing] = convert_track_pdgid_to_pid(
                pdgid_arr=chunk['track_pdgid'], # type: ignore
            )

            ###################################################################
            # tower
            # photon first
            ###################################################################
            for feature in ['pt', 'eta', 'phi']:
                arrays = [
                    ak.values_astype(
                        array=chunk[f'{prefix}_{feature}'],  # type: ignore
                        to=np.float32
                    )
                    for prefix in ['photon', 'neutral_hadron']
                ]
                chunk[f'tower_{feature}'] = ak.concatenate( # type: ignore
                    arrays=arrays,
                    axis=1
                )

            tower_pt = chunk['tower_pt'] # type: ignore
            tower_phi = chunk['tower_phi'] # type: ignore
            output['tower_px'][slicing] = tower_pt * np.cos(tower_phi)
            output['tower_py'][slicing] = tower_pt * np.sin(tower_phi)
            output['tower_eta'][slicing] = chunk['tower_eta'] # type: ignore

            output['tower_pid'][slicing] = ak.concatenate(
                arrays=[
                    ak.full_like(
                        array=chunk['photon_pt'],  # type: ignore
                        fill_value=ParticleTypeIndex.PHOTON.value,
                        dtype=np.int64
                    ),
                    ak.full_like(
                        array=chunk['neutral_hadron_pt'], # type: ignore
                        fill_value=ParticleTypeIndex.NEUTRAL_HADRON.value,
                        dtype=np.int64
                    ),

                ],
                axis=1,
            )

            ###################################################################
            #
            ###################################################################
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


def run_test(
    input_dir_path: Path,
    output_file_path: Path,
    max_input_files: int | None,
) -> None:
    """
    """
    input_file_path_list = list(input_dir_path.glob('*.root'))
    input_file_path_list = input_file_path_list[:max_input_files]

    run(
        input_file_path_list=input_file_path_list,
        output_file_path=output_file_path
    )



def run_batch(
    dataset: str,
    counter: int,
    delphes_config: str,
) -> None:
    """
    """

    counter_label = f'{counter:06d}'

    input_dir = PROJECT_DATA_DIR / 'delphes' / delphes_config / dataset / counter_label
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    input_file_path_list = sorted(
        input_dir.glob('*.root'),
        key=lambda item: int(item.stem.removeprefix('output_'))
    )

    output_dir = PROJECT_DATA_DIR / 'ntuple' / delphes_config / dataset
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_file_path = output_dir / counter_label
    output_file_path = output_file_path.with_suffix('.h5')

    run(
        input_file_path_list=input_file_path_list,
        output_file_path=output_file_path
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers()


    test_parser = subparsers.add_parser(
        name='test',
        help='test mode',
    )
    test_parser.set_defaults(func=run_test)
    test_parser.add_argument(
        '-i', '--input-dir-path', type=Path, required=True,
        help='input directory',
    )
    test_parser.add_argument(
        '-o', '--output-file-path', type=Path, required=True,
        help='output file',
    )
    test_parser.add_argument(
        '-m', '--max-input-files', default=10, type=int,
        help='output file',
    )

    batch_parser = subparsers.add_parser(
        name='batch',
        help='batch mode',
    )
    batch_parser.set_defaults(func=run_batch)
    batch_parser.add_argument(
        '-d', '--dataset', type=str, required=True,
        help='dataset')
    batch_parser.add_argument('-c', '--counter', type=int, required=True,
                              help='counter')
    batch_parser.add_argument('--delphes-config', type=str,
                              default='CMS_PhaseII_200PU_v04',
                              help='delphes config')

    args = parser.parse_args()
    args = vars(args)

    func = args.pop('func')
    func(**args)


if __name__ == '__main__':
    args = main()
