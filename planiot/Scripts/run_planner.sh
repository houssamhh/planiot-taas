#! /bin/bash

domainFile=$1
problemFile=$2
solutionFile=$3

CMD="ff -o ${domainFile} -f ${problemFile}"

$CMD > ${solutionFile}

