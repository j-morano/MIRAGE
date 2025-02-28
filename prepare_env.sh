#!/bin/bash

################################################################################
# prepare_env.sh
#
# This script prepares the environment for the project
# It receives two arguments:
# 1. The weights to download
#   - 'all' for all weights (default)
#   - 'base' for the base model weights
#   - 'large' for the large model weights
#   - 'none' for no weights
# 2. The datasets to download
#   - 'all' for all datasets (default)
#   - 'classification' for the classification datasets
#   - 'segmentation' for the segmentation datasets
#   - 'none' for no datasets
#   - 'classification-non-cross' for the classification datasets without the
#       cross-dataset evaluation datasets.
#   - 'segmentation-non-cross' for the segmentation datasets without the
#       cross-dataset evaluation datasets.
# 3. (optional) nodelete: if present, the downloaded files will not be deleted
# 4. (optional) ignorepython: if present, the script will not check the python
#      version and will not download python 3.10.16.
# 5. (optional) --help | -h: print the usage and exit

# Check if the script is being run with the correct number of arguments
# If not, print the usage and exit


################################################################################
# Parse the arguments

function print_usage() {
    echo 'Usage: ./prepare_env.sh <weights> <datasets> [nodelete] [ignorepython]'
    echo '  <weights>: base | large | all | none'
    echo '  <datasets>: classification | segmentation | all | none'
    echo '  [nodelete]: if present, the downloaded files will not be deleted'
    echo '  [ignorepython]: if present, the script will not check the python version'
    exit
}


# Set the default values
nodelete=false
ignorepython=false
weights=all
datasets=all


if [ $# -gt 4 ]; then
    print_usage
fi
if [ $1 ]; then
    if [ $1 == '--help' ] || [ $1 == '-h' ]; then
        print_usage
    elif [ $1 == 'base' ] || [ $1 == 'large' ] || [ $1 == 'all' ] || [ $1 == 'none' ]; then
        weights=$1
    else
        print_usage
    fi
fi
if [ $2 ]; then
    if [ $2 == 'classification' ] || [ $2 == 'segmentation' ] || [ $2 == 'all' ] \
        || [ $2 == 'none' ] || [ $2 == 'classification-non-cross' ] \
        || [ $2 == 'segmentation-non-cross' ]; then
        datasets=$2
    else
        print_usage
    fi
fi
if [ $3 ]; then
    if [ $3 == 'nodelete' ]; then
        nodelete=true
    elif [ $3 == 'ignorepython' ]; then
        ignorepython=true
    else
        print_usage
    fi
fi
if [ $4 ]; then
    if [ $4 == 'nodelete' ]; then
        nodelete=true
    elif [ $4 == 'ignorepython' ]; then
        ignorepython=true
    else
        print_usage
    fi
fi


# Print the arguments
echo '⚙️ Running with the following arguments:'
echo '   📦 weights: '$weights
echo '   🩻 datasets: '$datasets
echo '   ⛔ nodelete: '$nodelete
echo '   🐍 ignorepython: '$ignorepython



################################################################################
# Constants


readonly TDIV='================================================================'
readonly BDIV='----------------------------------------------------------------'

readonly BASE_URL='https://github.com/j-morano/MIRAGE/releases/download'

################################################################################
# Functions


function step() {
    # Args:
    # - $1: text to print
    echo ''
    echo $TDIV
    echo $1'...'
    echo $BDIV
}


function download() {
    # Args:
    #  - $1: URL to download
    echo '  🔗 URL: '$1
    curl -L -O $1
}


check_files_starting_with() {
    local dir="$1"
    local prefix="$2"

    # Check if the directory exists
    if [ ! -d "$dir" ]; then
        echo "Directory '$dir' does not exist."
        return 1
    fi

    # Loop through files in the directory
    for file in "$dir"/*; do
        # Check if the file is a regular file and starts with the prefix
        if [[ "$(basename "$file")" == "$prefix"* ]]; then
            # echo "File '$file' starts with '$prefix'."
            return 0
        fi
    done

    # echo "No file in '$dir' starts with '$prefix'."
    return 1
}


function download_to_dir() {
    # Args:
    #  - $1: URL to download
    #  - $2: directory to save the file
    #  Check if '_part_a' in the name, if so, get the prefix before '_part_a',
    #    otherwise, get the prefix before '.'
    local prefix=''
    if [[ $1 == *_part_a* ]]; then
        prefix=$(basename ${1%_part_a*})
    else
        prefix=$(basename ${1%.*})
    fi
    if check_files_starting_with $2 $prefix; then
        echo '  📥 "'$(basename $1)'" already downloaded'
        return
    fi
    download $1
    mkdir -p $2
    mv $(basename $1) $2
}


function extract_files() {
    export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
    # Args:
    # - $1: directory containing the zip files
    for file in $1/*.zip; do
        if [ -f $file ]; then
            echo '  📦 Extracting '$file
            unzip -q $file -d $dir
            if [ ! $3 ]; then
                rm $file
            fi
        fi
    done
}


################################################################################
# Main


step '🐍 Creating and activating virtual environment'

# Check python version
version=$(python --version | cut -d ' ' -f 2)
echo "  🐍 System Python version: $version"
# If python version does not start with '3.10', download and install python 3.10
if [[ ! $version == 3.10* ]] && [ ! $ignorepython ]; then
    if [ -d Python-3.10.16 ]; then
        echo '  📥 Python 3.10.16 already downloaded'
    else
        echo '  ⚠️ The version of Python installed is not 3.10.x'
        # Ask the user if they want to download Python 3.10.16
        # Wait to press enter
        read -p '  Do you want to download and install Python 3.10.16? (y/n): ' user_input
        echo ''
        if [[ ! $user_input == 'y' ]]; then
            echo '  Exiting...'
            exit
        fi
        echo '  📥 Downloading Python 3.10.16...'
        download https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tgz
        tar -xvf Python-3.10.16.tgz
        if [ ! $3 ]; then
            rm Python-3.10.16.tgz
        fi
        cd Python-3.10.16
        ./configure --enable-optimizations
        make
        cd ..
    fi
    if [ -d venv ]; then
        echo '  🐉 Python environment already exists'
    else
        echo '  🐉 Creating Python environment...'
        ./Python-3.10.16/python -m venv venv
    fi
else
    if [ -d venv ]; then
        echo '  🐉 Python environment already exists'
    else
        echo '  🐉 Creating Python environment...'
        python -m venv venv
    fi
fi


source venv/bin/activate


step '📦 Upgrading pip'
pip install --upgrade pip


step '📋 Installing requirements'
pip install -r requirements.txt


step '🔚 Exiting virtual environment'
deactivate


step '📥 Downloading model weights'
if [ $weights == 'base' ] || [ $weights == 'all' ]; then
    echo '  🏝️ MIRAGE-Base'
    download_to_dir $BASE_URL'/weights/MIRAGE-Base.pth' __weights
fi
if [ $weights == 'large' ] || [ $weights == 'all' ]; then
    echo '  🏝️ MIRAGE-Large'
    download_to_dir $BASE_URL'/weights/MIRAGE-Large.pth' __weights
fi


step '📥 Downloading datasets'
if [ $datasets == 'classification' ] || [ $datasets == 'all' ] || [ $datasets == 'classification-non-cross' ]; then
    echo '  📊 Classification datasets'
    dir='__datasets/Classification'
    download_to_dir $BASE_URL'/cls-data/Duke_iAMD.zip' $dir
    download_to_dir $BASE_URL'/cls-data/GAMMA.zip' $dir
    download_to_dir $BASE_URL'/cls-data/Harvard_Glaucoma.zip' $dir
    download_to_dir $BASE_URL'/cls-data/Noor_Eye_Hospital.zip' $dir
    download_to_dir $BASE_URL'/cls-data/OCTDL.zip' $dir
    download_to_dir $BASE_URL'/cls-data/OCTID.zip' $dir
    download_to_dir $BASE_URL'/cls-data/OLIVES.zip' $dir
    if [ $datasets != 'classification-non-cross' ]; then
        download_to_dir $BASE_URL'/cls-data/Noor_Eye_Hospital_cross_train.zip' $dir
        download_to_dir $BASE_URL'/cls-data/Noor_Eye_Hospital_cross_test.zip' $dir
        download_to_dir $BASE_URL'/cls-data/UMN_Duke_Srinivasan_cross_test.zip' $dir
    fi
    extract_files $dir
fi
if [ $datasets == 'segmentation' ] || [ $datasets == 'all' ] || [ $datasets == 'segmentation-non-cross' ]; then
    echo ''
    echo '  🩻 Segmentation datasets'
    dir='__datasets/Segmentation'
    download_to_dir $BASE_URL'/seg-data/AROI.zip' $dir
    download_to_dir $BASE_URL'/seg-data/Duke_DME.zip' $dir
    if [ $datasets != 'segmentation-non-cross' ]; then
        download_to_dir $BASE_URL'/seg-data/Duke_iAMD_labeled_part_aa' $dir
        download_to_dir $BASE_URL'/seg-data/Duke_iAMD_labeled_part_ab' $dir
        download_to_dir $BASE_URL'/seg-data/Duke_iAMD_labeled_part_ac' $dir
        download_to_dir $BASE_URL'/seg-data/Duke_iAMD_labeled_part_ad' $dir
        download_to_dir $BASE_URL'/seg-data/Duke_iAMD_labeled_part_ae' $dir
        download_to_dir $BASE_URL'/seg-data/Duke_iAMD_labeled_part_af' $dir
        download_to_dir $BASE_URL'/seg-data/Duke_iAMD_labeled_part_ag' $dir
        download_to_dir $BASE_URL'/seg-data/Duke_iAMD_labeled_part_ah' $dir
        if [ ! -f $dir/Duke_iAMD_labeled.zip ] && [ ! -d $dir/Duke_iAMD_labeled ]; then
            echo '  🧩 Combining Duke_iAMD_labeled parts'
            cat $dir/Duke_iAMD_labeled_part_* > $dir/Duke_iAMD_labeled.zip
            if [ ! $3 ]; then
                rm $dir/Duke_iAMD_labeled_part_*
            fi
        fi
    fi
    download_to_dir $BASE_URL'/seg-data/GOALS.zip' $dir
    download_to_dir $BASE_URL'/seg-data/RETOUCH_part_aa' $dir
    download_to_dir $BASE_URL'/seg-data/RETOUCH_part_ab' $dir
    if [ ! -f $dir/RETOUCH.zip ] && [ ! -d $dir/RETOUCH ]; then
        echo '  🧩 Combining RETOUCH parts'
        cat $dir/RETOUCH_part_* > $dir/RETOUCH.zip
        if [ ! $3 ]; then
            rm $dir/RETOUCH_part_*
        fi
    fi
    extract_files $dir
fi


echo ''
echo '🎉 Environment ready!'
