#!/usr/bin/env python3

import argparse
from argparse import RawTextHelpFormatter
import os
from os.path import join, exists, basename, isdir, isfile
import subprocess
import urllib.request
import zipfile
from glob import glob



# Constants
TDIV = '=' * 64
BDIV = '-' * 64
BASE_URL = 'https://github.com/j-morano/MIRAGE/releases/download'



def step(text):
    print(f'\n{TDIV}')
    print(f'{text}...')
    print(BDIV)


def download(url, directory=None):
    print(f'  🔗 URL: {url}')
    local_filename = url.split('/')[-1]
    if directory:
        local_filename = join(directory, local_filename)
    urllib.request.urlretrieve(url, local_filename)
    return local_filename


def check_files_starting_with(directory, prefix, is_part=False):
    if not exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return False
    for file in os.listdir(directory):
        if file.startswith(prefix):
            if is_part:
                if isdir(join(directory, file)) or file == f"{prefix}.zip":
                    return True
            else:
                return True
    return False


def download_to_dir(url, directory):
    if '_part_a' in url:
        prefix = basename(url).split('_part_a')[0]
        is_part = True
    else:
        prefix = basename(url).split('.')[0]
        is_part = False
    if check_files_starting_with(directory, prefix, is_part):
        print(f'  📥 "{basename(url)}" already downloaded')
        return
    downloaded_file = download(url)
    os.makedirs(directory, exist_ok=True)
    os.rename(downloaded_file, join(directory, basename(downloaded_file)))


def remove(file_pattern):
    for file in glob(file_pattern):
        if isfile(file):
            print(f'  🗑️ Removing file {file}')
            os.remove(file)


def extract_files(directory, nodelete):
    for file in glob(join(directory, '*.zip')):
        print(f'  📦 Extracting {file}')
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(directory)
        if not nodelete:
            remove(file)


def join_dataset(directory, dataset, nodelete):
    if not exists(join(directory, f"{dataset}.zip")) and not exists(join(directory, dataset)):
        print(f'  🧩 Combining {dataset} parts')
        with open(join(directory, f"{dataset}.zip"), 'wb') as outfile:
            for part in sorted(glob(join(directory, f"{dataset}_part_*"))):
                with open(part, 'rb') as infile:
                    outfile.write(infile.read())
        if not nodelete:
            print(f'  🗑️ Removing {dataset} parts')
            for part in glob(join(directory, f"{dataset}_part_*")):
                os.remove(part)


def get_args():
    parser = argparse.ArgumentParser(
        description='Prepare environment for the project.',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        '-w', '--weights',
        choices=['base', 'large', 'all', 'none'],
        default='all',
        help='Which weights to download.\n'
        '  - all: download all weights (default)\n'
        '  - base: only download the base model weights\n'
        '  - large: only download the large model weights\n'
        '  - none: do not download any weights'
    )
    parser.add_argument(
        '-d', '--datasets',
        choices=[
            'classification',
            'segmentation',
            'all',
            'none',
            'classification-non-cross',
            'segmentation-non-cross'
        ],
        default='all',
        help='Which datasets to download.\n'
        '  - all: download all datasets (default)\n'
        '  - classification: only download the classification datasets\n'
        '  - segmentation: only download the segmentation datasets\n'
        '  - none: do not download any dataset\n'
        '  - classification-non-cross: download the classification datasets'
            ' without the cross-dataset evaluation datasets\n'
        '  - segmentation-non-cross: download the segmentation datasets without'
            ' the cross-dataset evaluation datasets'
    )
    parser.add_argument(
        '--nodelete', action='store_true',
        help='Do not delete downloaded files'
    )
    parser.add_argument(
        '--ignorepython', action='store_true',
        help='Ignore Python version check'
    )
    args = parser.parse_args()
    return args


def run_cmd(command: str):
    command_list = command.split()
    return subprocess.run(command_list, check=True)


def main():
    args = get_args()

    print('⚙️ Running with the following arguments:')
    print(f'   📦 weights: {args.weights}')
    print(f'   🩻 datasets: {args.datasets}')
    print(f'   ⛔ nodelete: {args.nodelete}')
    print(f'   🐍 ignorepython: {args.ignorepython}')

    step('🐍 Creating and activating virtual environment')
    version = subprocess.check_output(['python', '--version']).decode('utf-8').strip().split(' ')[1]
    print(f"  🐍 System Python version: {version}")
    if not version.startswith('3.10') and not args.ignorepython:
        if exists('Python-3.10.16'):
            print('  📥 Python 3.10.16 already downloaded')
        else:
            print('  ⚠️ The version of Python installed is not 3.10.x')
            user_input = input('  Do you want to download and install Python 3.10.16? (y/n): ')
            if user_input.lower() != 'y':
                print('  🚫 Skipping Python installation')
            else:
                print('  📥 Downloading Python 3.10.16...')
                download('https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tgz')
                run_cmd('tar -xvf Python-3.10.16.tgz')
                remove('Python-3.10.16.tgz')
                os.chdir('Python-3.10.16')
                run_cmd('./configure --enable-optimizations')
                run_cmd('make')
                os.chdir('..')
        if exists('venv'):
            print('  🐍 Python environment (Python 3.10.16) already exists')
        else:
            print('  🐍 Creating Python environment using Python 3.10.16...')
            run_cmd('./Python-3.10.16/python -m venv venv')
    else:
        if exists('venv'):
            print('  🐍 Python environment already exists')
        else:
            print('  🐍 Creating Python environment...')
            run_cmd('python -m venv venv')

    step('📦 Upgrading pip')
    run_cmd('./venv/bin/pip install --upgrade pip')

    step('📋 Installing requirements')
    run_cmd('./venv/bin/pip install -r requirements.txt')

    step('📥 Downloading model weights')
    if args.weights in ['base', 'all']:
        print('  🏝️ MIRAGE-Base')
        download_to_dir(f'{BASE_URL}/weights/MIRAGE-Base.pth', '__weights')
    if args.weights in ['large', 'all']:
        print('  🏝️ MIRAGE-Large')
        download_to_dir(f'{BASE_URL}/weights/MIRAGE-Large.pth', '__weights')

    step('📥 Downloading datasets')
    if args.datasets in ['classification', 'all', 'classification-non-cross']:
        print('  📊 Classification datasets')
        dir = '__datasets/Classification'
        download_to_dir(f'{BASE_URL}/cls-data/Duke_iAMD.zip', dir)
        download_to_dir(f'{BASE_URL}/cls-data/GAMMA.zip', dir)
        download_to_dir(f'{BASE_URL}/cls-data/Harvard_Glaucoma.zip', dir)
        download_to_dir(f'{BASE_URL}/cls-data/Noor_Eye_Hospital.zip', dir)
        download_to_dir(f'{BASE_URL}/cls-data/OCTDL.zip', dir)
        download_to_dir(f'{BASE_URL}/cls-data/OCTID.zip', dir)
        download_to_dir(f'{BASE_URL}/cls-data/OLIVES.zip', dir)
        if args.datasets != 'classification-non-cross':
            download_to_dir(f'{BASE_URL}/cls-data/Noor_Eye_Hospital_cross_train.zip', dir)
            download_to_dir(f'{BASE_URL}/cls-data/Noor_Eye_Hospital_cross_test.zip', dir)
            download_to_dir(f'{BASE_URL}/cls-data/UMN_Duke_Srinivasan_cross_test.zip', dir)
        extract_files(dir, nodelete=args.nodelete)
    if args.datasets in ['segmentation', 'all', 'segmentation-non-cross']:
        print('  🩻 Segmentation datasets')
        dir = '__datasets/Segmentation'
        download_to_dir(f'{BASE_URL}/seg-data/AROI.zip', dir)
        download_to_dir(f'{BASE_URL}/seg-data/Duke_DME.zip', dir)
        if args.datasets != 'segmentation-non-cross':
            for part in ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah']:
                download_to_dir(f'{BASE_URL}/seg-data/Duke_iAMD_labeled_part_{part}', dir)
            join_dataset(dir, 'Duke_iAMD_labeled', nodelete=args.nodelete)
        download_to_dir(f'{BASE_URL}/seg-data/GOALS.zip', dir)
        for part in ['aa', 'ab']:
            download_to_dir(f'{BASE_URL}/seg-data/RETOUCH_part_{part}', dir)
        join_dataset(dir, 'RETOUCH', nodelete=args.nodelete)
        extract_files(dir, nodelete=args.nodelete)

    print('\n🎉 Environment ready!')



if __name__ == '__main__':
    main()
