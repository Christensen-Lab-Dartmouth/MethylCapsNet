FROM joshualevy44/methylnet:latest

RUN mkdir ./methylcaps_code

COPY . ./methylcaps_code/

RUN conda install -y -c bioconda bedtools

RUN pip3 install --upgrade setuptools

RUN apt install git-lfs

RUN git clone https://github.com/Christensen-Lab-Dartmouth/MethylCapsNet && cd MethylCapsNet && pip install . && install_methylcaps_dependencies

RUN cp -al ./methylcaps_code/test_data/* /pymethyl/

RUN chmod 755 -R /pymethyl/test_data/

WORKDIR /pymethyl
