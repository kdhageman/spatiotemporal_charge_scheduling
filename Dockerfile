FROM continuumio/miniconda3

WORKDIR /root

RUN apt-get update
RUN apt install -y gcc g++ ffmpeg libsm6 libxext6 curl

RUN pip install pyyaml jsons networkx Cython open3d pyomo simpy PyPDF2
RUN conda install -y numpy matplotlib
RUN conda install -y -c conda-forge cvxpy

# install networkx-metis
RUN git clone https://github.com/networkx/networkx-metis.git && cd networkx-metis && python setup.py install && cd ../ && rm -rf networkx-metsi

# install Gurobi
RUN conda config --add channels https://conda.anaconda.org/gurobi && conda -y install gurobi

RUN echo export 'PYTHONPATH=/project' >> /root/.bashrc
RUN touch /root/gurobi.lic

WORKDIR /project

ENTRYPOINT /bin/bash