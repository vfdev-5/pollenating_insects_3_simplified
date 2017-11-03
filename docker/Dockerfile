FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER vfdev-5

LABEL version="0.1"
LABEL description="Pollenating Insects 3 Challenge (Simplified)"


# Update all
RUN apt-get update


# Setup en_US locale
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# Install dependencies:
RUN apt-get install -y --no-install-recommends \
	build-essential \
	curl \
	wget \
	python \
	python-dev \
	python-pip \
	graphviz \
	unzip \
	screen \
	nano \
	git


RUN apt-get clean && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip


# Install python packages
RUN pip install --no-cache-dir setuptools


# Install miniconda
RUN wget -q http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh -O $HOME/miniconda.sh && \
     chmod +x $HOME/miniconda.sh && \
     $HOME/miniconda.sh -b -p $HOME/miniconda


ENV PATH /root/miniconda/bin:$PATH


RUN conda update --yes --quiet conda
RUN conda install --yes numpy
RUN conda install --yes pandas
RUN conda install --yes scikit-learn
RUN conda install --yes scikit-image
RUN conda install --yes joblib
RUN conda install --yes cloudpickle
RUN conda install --yes gitpython


RUN pip install h5py==2.7.1 \
     tensorflow-gpu==1.3.0 \
     keras==2.0.8


RUN cd $HOME && git clone https://github.com/ramp-kits/pollenating_insects_3_simplified

RUN cd $HOME && git clone https://github.com/paris-saclay-cds/ramp-workflow && cd ramp-workflow && \
     python setup.py install


## Custom dependencies

RUN pip install opencv-python

RUN pip install git+git://github.com/aleju/imgaug/ \
                git+git://github.com/farizrahman4u/keras-contrib