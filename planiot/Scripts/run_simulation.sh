#! /bin/bash

ARGS=$*

CLASSPATH=Code/PlanIoT-SEAMS2023/queueingNetworkComposer/target/dependency/*:Code/PlanIoT-SEAMS2023/queueingNetworkComposer/target/classes

CLASS=composer.Main

# Start the composer
CMD="java -cp ${CLASSPATH} ${CLASS} ${ARGS}"

$CMD
