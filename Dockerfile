FROM continuumio/miniconda3

RUN apt-get update
RUN apt install -y gcc g++ ffmpeg libsm6 libxext6 curl

RUN pip install pyyaml jsons networkx Cython open3d pyomo simpy PyPDF2
RUN conda install -y numpy matplotlib
RUN conda install -c conda-forge cvxpy

# install networkx-metis
RUN git clone https://github.com/networkx/networkx-metis.git && cd networkx-metis && python setup.py install

# install Gurobi
RUN curl https://packages.gurobi.com/9.5/gurobi9.5.2_linux64.tar.gz  --output gurobi.tar.gz && tar xvfz gurobi.tar.gz -C /opt && echo 'export PATH="${PATH}:/opt/gurobi952/linux64/bin"' >> /root/.bashrc

RUN echo export 'PYTHONPATH=/project' >> /root/.bashrc

WORKDIR /project

ENTRYPOINT /bin/bash