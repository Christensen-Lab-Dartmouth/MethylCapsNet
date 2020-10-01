FROM joshualevy44/methylnet:0.1

RUN mkdir ./methylcaps_code

COPY . ./methylcaps_code/

RUN apt install -y bedtools git-lfs

RUN git clone https://github.com/Christensen-Lab-Dartmouth/MethylCapsNet 

RUN cd MethylCapsNet && pip install . && install_methylcaps_dependencies && cd .

RUN cp -al ./methylcaps_code/test_data/* /pymethyl/

RUN chmod 755 -R /pymethyl/test_data/

WORKDIR /pymethyl
