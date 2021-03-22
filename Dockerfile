FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel 
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /workspace/

# general tools
RUN apt-get update -y
RUN apt-get install -y apt-transport-https bzip2 ca-certificates cmake curl ffmpeg g++ gcc git gnupg2 less
RUN apt-get install -y make sox vim sed

# install kubectl: this could be simplified to apt-get install when using devel. keeping for runtime
RUN curl -o apt-key.gpg "https://packages.cloud.google.com/apt/doc/apt-key.gpg"
RUN echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" >> /etc/apt/sources.list.d/kubernetes.list
RUN apt-key add apt-key.gpg
RUN rm apt-key.gpg
RUN apt-get update
RUN apt-get install -y kubectl

# install google-cloud-sdk: this could be simplified to apt-get install when using devel. keeping for runtime
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" >>  /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl -o gcsdk-apt-key.gpg "https://packages.cloud.google.com/apt/doc/apt-key.gpg"
RUN apt-key --keyring /usr/share/keyrings/cloud.google.gpg add gcsdk-apt-key.gpg
RUN rm gcsdk-apt-key.gpg
RUN apt-get update
RUN apt-get install -y google-cloud-sdk

# install warp-CTC
ENV CUDA_HOME=/usr/local/cuda
RUN git clone https://github.com/SeanNaren/warp-ctc.git warp-ctc-naren
RUN cd warp-ctc-naren
# these sed commands are needed to remove GPU requirements that don't allow the use of CUDA 11
RUN sed -i '35 s/./#&/' CMakeLists.txt
RUN sed -i '36 s/./#&/' CMakeLists.txt
RUN sed -i '37 s/./#&/' CMakeLists.txt
RUN sed -i '38 s/./#&/' CMakeLists.txt
RUN mkdir build; cd build; cmake ..; make
RUN cd warp-ctc; cd pytorch_binding; python setup.py install

#add the main repo
ADD . /workspace/

RUN pip install -r requirements_py38.txt


CMD python kubernetes/scripts/pod_stats.py 
#CMD sh hello.sh 
#CMD python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 train.py configs/ctc_config_ph0.yaml 
