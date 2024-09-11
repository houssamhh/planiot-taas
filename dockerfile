FROM ubuntu:20.04

# set default values for build arguments
ARG UNAME=planiot
ARG UID=1000
ARG GID=1000

# update and upgrade
RUN apt-get update
RUN apt-get -qy dist-upgrade

RUN apt-get install -qy --no-install-recommends \
    sudo \
    make gcc libc6-dev \
    wget \
    openjdk-11-jdk maven \
    python \
    flex bison

# install Metric-FF version 2.0
# https://fai.cs.uni-saarland.de/hoffmann/metric-ff.html
RUN cd /opt; \
    wget --progress dot:mega -O Metric-FF-v2.0.tgz https://fai.cs.uni-saarland.de/hoffmann/ff/Metric-FF-v2.0.tgz; \
    tar xfz Metric-FF-v2.0.tgz; \
    rm -f Metric-FF-v2.0.tgz; \
    echo "Changes of constant values in Metric-FF-v2.0/ff.h"; \
    sed --in-place 's/MAX_LNF_F 25/MAX_LNF_F 150/' Metric-FF-v2.0/ff.h; \
    grep 'MAX_LNF_F ' Metric-FF-v2.0/ff.h; \
    sed --in-place 's/MAX_LNF_EFFS 50/MAX_LNF_EFFS 200/' Metric-FF-v2.0/ff.h; \
    grep 'MAX_LNF_EFFS ' Metric-FF-v2.0/ff.h; \
    (cd Metric-FF-v2.0; make)

# install JMT version 1.2.2
# https://jmt.sourceforge.net/Download.html
RUN cd /opt; \
    mkdir JMT-v1.2.2; \
    cd JMT-v1.2.2; \
    wget --progress dot:mega -O jmt-singlejar-1.2.2.jar http://sourceforge.net/projects/jmt/files/jmt/JMT-1.2.2/JMT-singlejar-1.2.2.jar/download

# create user and configure .bashrc
RUN groupadd -g $GID -o $UNAME; \
    useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME; \
    echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" >> /home/$UNAME/.bashrc; \
    echo "PATH=\${PATH}:/opt/Metric-FF-v2.0/" >> /home/$UNAME/.bashrc; \
    echo "export CLASSPATH=/usr/lib/jvm/java-11-openjdk-amd64/lib:/opt/JMT-v1.2.2/jmt-singlejar-1.2.2.jar" >> /home/$UNAME/.bashrc; \
    cat /home/$UNAME/.bashrc; \
    mkdir -p /home/$UNAME/planiot

# set user, environment, and working directory
USER $UNAME
ENV HOME=/home/$UNAME
WORKDIR /home/$UNAME

ENTRYPOINT ["/bin/bash"]
