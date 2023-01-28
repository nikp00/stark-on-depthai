#!/bin/bash

usage () {
    echo """
    Usage: $0 [OPTIONS]

    OPTIONS:
        -l, --lasot <path to LASOT dataset>                 # Create link to LASOT dataset
        -v, --vot <path to VOT dataset>                     # Create link to VOT dataset
        --setup-stark-checkpoints <path to checkpoints>     # Copy STARK checkpoints to correct location
        --make-env-file                                     # Create env file
        --make-python-venv                                  # Create python virtual environment
        -h help
    
    """
}

create_link () {
    path=$(readlink -f $1)

    if [ ! -d $path ]; then
        echo "Error: $path does not exist"
        exit 1
    else
        echo "Creating link to $path"
        ln -s $path $2
    fi

}

for arg in "$@"; do
    shift
    case $arg in
        --lasot)
            mkdir -p dataset
            create_link $1 "dataset/LASOT"
            shift
            ;;
        --vot)
            mkdir -p dataset
            create_link $1 "dataset/VOT"
            shift
            ;;
        --make-env-file)
            echo "Creating env file"
            echo "export BASE_PATH="$(pwd) > .env
            cat .env
            ;;
        --setup-stark-checkpoints)
            echo "Setting up STARK checkpoint"
            cp -r $1 Stark/checkpoints
            shift
            ;;
        --make-python-venv)
            echo "Creating python virtual environment"
            python3 -m venv venv
            source venv/bin/activate
            pip install -r requirements.txt
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done


