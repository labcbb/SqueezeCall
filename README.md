# SqueezeCall
SqueezeCall: Nanopore basecalling using a Squeezeformer network.  

In SqueezeCall, convolution layers are used to downsample raw signals. A Squeezeformer network is employed to capture global context. Finally, a CTC decoder generates the DNA sequence by a beam search algorithm. Experiments on multiple species further demonstrate the potential of the Squeezeformer-based model to improve basecalling accuracy and its superiority over a recurrent neural network (RNN)-based model and Transformer-based models.

## Install

``` sh
pip install -r requirements.txt
```

# Getting started

## Data download
Human raw data and basecall datasets (FAF04090, FAF09968, FAB42828) are available at https://github.com/nanopore-wgs-consortium/NA12878/blob/master/Genome.md; bacterial raw data and lambda phage data are available at https://github.com/marcpaga/nanopore_benchmark/tree/main/download; ONT chunk dataset are available at https://cdn.oxfordnanoportal.com/software/analysis/bonito/example_data_dna_r9.4.1_v0.zip.

## Demo data

The test data for this project can be downloaded from [10.5281/zenodo.14725006](https://zenodo.org/records/14725007)

## Data processing

There's two main data processing steps:

### Annotate the raw data with the reference sequence

```
tombo \
resquiggle demo_data/fast5 \
demo_data/Lambda_phage.fna \
--processes 2 \
--dna \
--num-most-common-errors 5 \
--ignore-read-locks
```

### Chunk the raw signal and save it numpy arrays

In this step, we take the raw signal and splice it into segments of a fixed length so that they can be fed into the neural network.

```
python ./data_prepare_numpy.py \
--fast5-dir  ./demo_data/fast5 \
--output-dir  ./demo_data/nn_input \
--total-files  1 \
--window-size 2000 \
--window-slide 0 \
--n-cores 2 \
--verbose
```

## Model training
``` sh
python train.py --config_path ./config/base.yaml --data_dir ./demo_data/nn_input
```

## Referenced the following GitHub
https://github.com/wenet-e2e/wenet  
https://github.com/nanoporetech/bonito  
https://github.com/marcpaga/basecalling_architectures?tab=readme-ov-file

