FROM opencog/opencog-deps:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN mkdir -p /opt/singnet/projects

WORKDIR /opt/singnet/projects/

# Cython and Jupyter
RUN pip3.8 install cython cytoolz jupyter

# Awakening python deps
RUN pip3.8 install amrlib word2number pandas

# Cogutil
RUN git clone https://github.com/opencog/cogutil.git
RUN cd cogutil/ && mkdir build && cd build && cmake ..
RUN cd cogutil/build && make -j5 install
RUN rm -rf cogutil

# AtomSpace
RUN git clone https://github.com/opencog/atomspace.git
RUN cd atomspace && mkdir build && cd build/ && cmake .. && make -j5
RUN cd atomspace/build && make -j5 install
RUN rm -rf atomspace/build

# URE
RUN git clone https://github.com/opencog/ure.git
RUN cd ure && mkdir build && cd build/ && cmake .. && make -j5
RUN cd ure/build && make -j5 install
RUN rm -rf ure/build

# Miner
RUN git clone https://github.com/opencog/miner.git
RUN cd miner && mkdir build && cd build/ && cmake .. && make -j5
RUN cd miner/build && make -j5 install
RUN rm -rf miner/build

# CogServer
RUN git clone https://github.com/opencog/cogserver.git
RUN cd cogserver && mkdir build && cd build/ && cmake .. && make -j5
RUN cd cogserver/build && make -j5 install
RUN rm -rf cogserver/build

# Attention
RUN git clone https://github.com/opencog/attention.git
RUN cd attention && mkdir build && cd build/ && cmake .. && make -j5
RUN cd attention/build && make -j5 install
RUN rm -rf attention/build

# SpaceTime
RUN git clone https://github.com/opencog/spacetime.git
RUN cd spacetime && mkdir build && cd build/ && cmake .. && make -j5
RUN cd spacetime/build && make -j5 install
RUN rm -rf spacetime/build

# PLN
RUN git clone https://github.com/opencog/pln.git
RUN cd pln && mkdir build && cd build/ && cmake .. && make -j5
RUN cd pln/build && make -j5 install
RUN rm -rf pln/build

# LG-Atomese
RUN git clone https://github.com/opencog/lg-atomese.git
RUN cd lg-atomese && mkdir build && cd build/ && cmake .. && make -j5
RUN cd lg-atomese/build && make -j5 install
RUN rm -rf lg-atomese/build

# OpenCog
RUN git clone https://github.com/opencog/opencog.git
RUN cd opencog && mkdir build && cd build/ && cmake .. && make -j5
RUN cd opencog/build && make -j5 install
RUN rm -rf opencog/build

RUN echo "export GUILE_AUTO_COMPILE=0" >> ~/.profile

RUN echo "(use-modules (ice-9 readline)) (activate-readline)\
(add-to-load-path \"/usr/local/share/opencog/scm\")\
(add-to-load-path \"/opt/singnet/projects/opencog/examples/pln/conjunction/\")\
(add-to-load-path \"/opt/singnet/projects/atomspace/examples/rule-engine/rules/\")\
(add-to-load-path \"/opt/singnet/projects/opencog/opencog/pln/rules/\")\
(add-to-load-path \".\")\
(use-modules (opencog))\
(use-modules (opencog query))\
(use-modules (opencog exec))" >> ~/.guile

RUN echo "export PYTHONPATH=\$PYTHONPATH:/usr/local/lib/python3.8/dist-packages/" >> ~/.profile
RUN echo "export PYTHONPATH=\$PYTHONPATH:/usr/local/lib/python3/dist-packages/" >> ~/.profile

CMD jupyter notebook --ip=0.0.0.0 --allow-root
