#!/usr/bin/env python
import argparse
import subprocess
from subprocess import PIPE
from pathlib import Path


def run(
    dataset: str,
    counter_list: list[int],
) -> None:
    """
    """

    prefix = Path(__file__).parent
    script = prefix / 'job.fish'
    if not script.exists():
        raise FileNotFoundError(script)

    for counter in counter_list:
        print(f'{counter=}')
        args = [
            'sbatch',
            f'--export=dataset={dataset},counter={counter}',
            script,
        ]
        args = list(map(str, args))
        result = subprocess.run(args, stdout=PIPE, stderr=PIPE)
        if len(result.stderr) > 0:
            raise RuntimeError(result.stderr)

        print(result.stdout.decode('utf-'))
        print('-' * 80)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default='TTto2L2Nu-2Jets_madgraphMLM-pythia8',
        help='dataset',
    )
    parser.add_argument(
        '-c', '--counter-list',
        type=int,
        nargs='+',
        required=True,
        help='list of counters',
    )

    args = parser.parse_args()

    run(**vars(args))


if __name__ == '__main__':
    main()
