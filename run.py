#!/usr/bin/env python
"""
"""
import argparse
from enum import Enum
from typing import Final, cast
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
import uproot.writing
import h5py as h5
import vector
from vector import MomentumNumpy2D
import tqdm

vector.register_awkward()


# FIXME:
PROJECT_DATA_DIR = Path('/store/hep/users/slowmoyang/diffmet')


BRANCH_DICT: Final[dict[str, list[str]]] = {
    'PFCandidate': [
        'Px', 'Py', 'Eta', 'E', 'D0', 'DZ', 'PUPPIWeight',
        'IsRecoPU', 'PID', 'Charge',
        'GenLVCount', 'GenLVPx', 'GenLVPy', 'GenLVPz', 'GenLVEnergy',
        'GenPUCount', 'GenPUPx', 'GenPUPy', 'GenPUPz', 'GenPUEnergy',
    ],
    'LVGenMET': ['MET', 'Phi'],
    'Weight': ['Weight'],
}


BRANCH_LIST: Final[list[str]] = [
    f'{branch}/{branch}.{leaf}'
    for branch, leaf_list in BRANCH_DICT.items()
    for leaf in leaf_list
]


PREFIX_ALIAS_DICT: Final[dict[str, str]] = {
    'PFCandidate': 'pf',
    'LVGenMET': 'gen_met',
    'Weight': 'gen',
}

FEATURE_ALIAS_DICT: Final[dict[str, str]] = {
    'MET': 'PT',
    'IsRecoPU': 'is_reco_pu',
    'PID': 'pdgid',
    'PUPPIWeight': 'puppi_weight',
    'E': 'energy',
}


def make_alias(
    branch: str
) -> str:
    """
    """
    prefix, feature = branch.split('/')[1].split('.')
    prefix = PREFIX_ALIAS_DICT[prefix]
    feature = FEATURE_ALIAS_DICT.get(feature, feature)
    # FIXME:
    if feature.startswith('GenLV'):
        feature = feature.replace('GenLV', 'gen_lv_')
    elif feature.startswith('GenPU'):
        feature = feature.replace('GenPU', 'gen_pu_')
    else:
        ...
    feature = feature.lower()
    alias = f'{prefix}_{feature}'
    return alias


class VertexType(Enum):
    ZERO_PAD = 0
    NEUTRAL = 1
    CHARGED_RECO_PU = 2
    CHARGED_NOT_RECO_PU = 3


class ParticleType(Enum):
    ZERO_PAD = 0
    ELECTRON = 1
    MUON = 2
    CHARGED_HADRON = 3
    PHOTON = 4
    NEUTRAL_HADRON = 5


class ChargeType(Enum):
    ZERO_PAD = 0
    NEGATIVE = 1
    NEUTRAL = 2
    POSITIVE = 3


PDGID_TO_PID: Final[dict[int, int]] = {
    11: ParticleType.ELECTRON.value,
    13: ParticleType.MUON.value,
    22: ParticleType.PHOTON.value,
    0: ParticleType.NEUTRAL_HADRON.value,
}


def _make_particle_type(
    pdgid_arr: ak.Array
) -> np.ndarray:
    """
    """
    pdgid_np_arr = ak.to_numpy(np.abs(pdgid_arr))
    pid_arr = np.full_like(
        a=pdgid_np_arr,
        fill_value=ParticleType.CHARGED_HADRON.value,
    )
    for pdgid, pid in PDGID_TO_PID.items():
        mask = pdgid_arr == pdgid
        pid_arr[mask] = pid
    return pid_arr


def make_particle_type(
    pdgid_chunk: ak.Array
) -> ak.Array:
    """
    """
    return ak.Array(list(map(_make_particle_type, pdgid_chunk)))


def make_charge_type(
    charge_chunk: ak.Array,
) -> ak.Array:
    """
    """
    charge_type_chunk = charge_chunk + 2
    return charge_type_chunk


def _make_vertex_type(
    charge_arr: ak.Array,
    is_reco_pu_arr: ak.Array,
) -> np.ndarray:
    """
    """
    charged_mask: np.ndarray = ak.to_numpy(charge_arr != 0) # type: ignore
    pu_mask: np.ndarray = ak.to_numpy(is_reco_pu_arr == 1) # type: ignore

    output = np.full_like(
        a=pu_mask,
        fill_value=VertexType.NEUTRAL.value,
        dtype=np.int64,
    )
    output[charged_mask & pu_mask] = VertexType.CHARGED_RECO_PU.value
    output[charged_mask & ~pu_mask] = VertexType.CHARGED_NOT_RECO_PU.value
    return output


def make_vertex_type(
    charge_chunk: ak.Array,
    is_reco_pu_chunk: ak.Array,
) -> ak.Array:
    return ak.Array(list(map(_make_vertex_type, charge_chunk, is_reco_pu_chunk)))


def reconstruct_met(
    particle_arr_chunk: list[MomentumNumpy2D]
) -> MomentumNumpy2D:
    """
    """
    met_chunk = [each.sum().neg2D for each in particle_arr_chunk]
    return MomentumNumpy2D(met_chunk) # type: ignore


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
    total = sum(num_entries
                for *_, num_entries
                in uproot.num_entries(input_files))

    output_file = h5.File(output_file_path, 'w')

    def create_dataset(
        name: str,
        dtype: type = np.float32,
        vlen: bool = False,
    ) -> h5.Dataset:
        """create variable-length dataset"""
        if vlen:
            dtype = h5.vlen_dtype(dtype)
        return output_file.create_dataset(name=name, shape=total, dtype=dtype)

    output: dict[str, h5.Dataset] = {}

    ###########################################################################
    # createt datasets for PF objects
    ###########################################################################
    # NOTE: float32
    pf_float32_suffix_list = [
        'px', 'py', 'eta', 'energy', 'd0', 'dz', 'puppi_weight',
    ]
    pf_float32_suffix_list += [
        f'gen_{vertex_type}_{feature}'
        for vertex_type in ['lv', 'pu']
        for feature in ['px', 'py', 'pz', 'energy']
    ]
    # NOTE: int64
    pf_int64_suffix_list = [
        'is_reco_pu', 'charge', 'pdgid',
        'gen_lv_count', 'gen_pu_count',
    ]

    extra_list = [
        'vertex_type',
        'particle_type',
        'charge_type',
    ]

    pf_suffix_list = pf_float32_suffix_list + pf_int64_suffix_list + extra_list

    # NOTE:
    for suffix in pf_suffix_list:
        key = f'pf_{suffix}'
        dtype = np.float32 if suffix in pf_float32_suffix_list else np.int64
        output[key] = create_dataset(key, dtype, vlen=True)

    ###########################################################################
    # create datasets for various types of METs
    ###########################################################################
    for obj in ['gen_met', 'pf_met', 'puppi_met']:
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
            aliases=aliases,
            library='ak',
        )
        for chunk in chunk_iterator:
            chunk = cast(dict[str, ak.Array], chunk)

            num_entries = len(chunk)
            start = stop
            stop = start + num_entries
            slicing = slice(start, stop)

            ####################################################################
            # NOTE: PF
            #
            ####################################################################
            for suffix in pf_float32_suffix_list + pf_int64_suffix_list:
                key = f'pf_{suffix}'
                output[key][slicing] = chunk[key]

            output[f'pf_vertex_type'][slicing] = make_vertex_type(
                charge_chunk=chunk['pf_charge'],
                is_reco_pu_chunk=chunk['pf_is_reco_pu'],
            )

            output['pf_particle_type'][slicing] = make_particle_type(
                pdgid_chunk=chunk['pf_pdgid'],
            )

            output['pf_charge_type'][slicing] = make_charge_type(
                charge_chunk=chunk['pf_charge'],
            )


            ###################################################################
            #
            ###################################################################
            for obj in ['gen_met']:
                for feature in ['pt', 'phi']:
                    key = f'{obj}_{feature}'
                    output[key][slicing] = ak.values_astype(
                        array=ak.flatten(chunk[key]),
                        to=np.float32
                    )

            particle_arr_dict: dict[str, list[MomentumNumpy2D]] = {}

            particle_arr_dict['pf'] = [ # type: ignore
                MomentumNumpy2D(dict(px=px, py=py))
                for px, py in zip(chunk['pf_px'], chunk['pf_py'])
            ]

            particle_arr_dict['puppi'] = [ # type: ignore
                MomentumNumpy2D(wgt * pf)
                for pf, wgt
                in zip(particle_arr_dict['pf'], chunk['pf_puppi_weight'])
            ]

            met_dict: dict[str, MomentumNumpy2D] = {
                key: reconstruct_met(particle_arr)
                for key, particle_arr in particle_arr_dict.items()
            }

            for prefix, met in met_dict.items():
                for feature in ['pt', 'phi']:
                    key = f'{prefix}_met_{feature}'
                    output[key][slicing] = ak.values_astype(
                        array=ak.flatten(getattr(met, feature)),
                        to=np.float32
                    )

            ###################################################################
            #
            ###################################################################
            for key in ['gen_weight']:
                array = ak.to_numpy(array=chunk[key][:, 0])
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

    ###########################################################################
    # NOTE: test mode
    #
    ###########################################################################
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

    ###########################################################################
    # NOTE: batch mdoe
    #
    ###########################################################################
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
