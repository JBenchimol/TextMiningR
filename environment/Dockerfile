# hash:sha256:c5badf42ab5c561a85e847547482cf355d2145aab7613f23a5694b55e87b5cdf
FROM registry.codeocean.com/codeocean/r-studio:1.4.1106-r4.0.5-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgsl0-dev=2.4+dfsg-6 \
        libpoppler-cpp-dev=0.62.0-2ubuntu2.12 \
    && rm -rf /var/lib/apt/lists/*

RUN Rscript -e 'remotes::install_version("NLP", "0.2-1")' \
    && Rscript -e 'remotes::install_version("SnowballC", "0.7.0")' \
    && Rscript -e 'remotes::install_version("antiword", "1.3")' \
    && Rscript -e 'remotes::install_version("cluster", "2.1.2")' \
    && Rscript -e 'remotes::install_version("data.table", "1.14.2")' \
    && Rscript -e 'remotes::install_version("devtools", "2.4.1")' \
    && Rscript -e 'remotes::install_version("dplyr", "1.0.8")' \
    && Rscript -e 'remotes::install_version("ggplot2", "3.3.5")' \
    && Rscript -e 'remotes::install_version("ggthemes", "4.2.4")' \
    && Rscript -e 'remotes::install_version("gplots", "3.1.1")' \
    && Rscript -e 'remotes::install_version("gsl")' \
    && Rscript -e 'remotes::install_version("igraph", "1.2.11")' \
    && Rscript -e 'remotes::install_version("lattice", "0.20-45")' \
    && Rscript -e 'remotes::install_version("modeltools", "0.2-23")' \
    && Rscript -e 'remotes::install_version("pheatmap", "1.0.12")' \
    && Rscript -e 'remotes::install_version("plyr", "1.8.6")' \
    && Rscript -e 'remotes::install_version("qpdf", "1.1")' \
    && Rscript -e 'remotes::install_version("quanteda", "3.2.1")' \
    && Rscript -e 'remotes::install_version("quanteda.textmodels", "0.9.4")' \
    && Rscript -e 'remotes::install_version("readODS", "1.7.0")' \
    && Rscript -e 'remotes::install_version("readtext", "0.81")' \
    && Rscript -e 'remotes::install_version("rowr", "1.1.3")' \
    && Rscript -e 'remotes::install_version("streamR", "0.4.5")' \
    && Rscript -e 'remotes::install_version("stringi", "1.7.6")' \
    && Rscript -e 'remotes::install_version("stringr", "1.4.0")' \
    && Rscript -e 'remotes::install_version("striprtf", "0.5.3")' \
    && Rscript -e 'remotes::install_version("tidyr", "1.2.0")' \
    && Rscript -e 'remotes::install_version("tm", "0.7-8")' \
    && Rscript -e 'remotes::install_version("topicmodels", "0.2-12")' \
    && Rscript -e 'remotes::install_version("wordcloud", "2.6")' \
    && Rscript -e 'remotes::install_version("xts", "0.12.1")'

RUN Rscript -e 'options(warn=2); install.packages("BiocManager")'
RUN Rscript -e 'options(warn=2); BiocManager::install(c( \
        "Rgraphviz" \
    ))' # Original versions: 2.38.0
