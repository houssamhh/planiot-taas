#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:54:37 2022

@author: houssam
"""

import csv
import os
import sys

def get_path_of_generated(path):
    normalized_path = os.path.normpath(path)
    path_parts = normalized_path.split(os.sep)
    filename = path_parts[-1]
    if ("template" in filename):
        generated_file_name = filename.replace("template", "generated")
    else:
        generated_file_name = filename.split('.')[0] + "-generated.pddl"
    path_parts.pop()
    path_parts.pop()
    output_file_path = os.sep.join(path_parts) + os.sep + 'pddl-files' + os.sep + generated_file_name
    return output_file_path
    
def instantiate_files(dataset_path, domain_template, problem_template, domain_generated, problem_generated):
    baseline_str = ''
    droppingVS1_str = ''
    droppingVS2_str = ''
    droppingVS2AN2_str = ''
    
    prioRT_str = ''
    prioAN_str = ''
    prioTS_str = ''
    prioVS_str = ''
    prioRTVS_str = ''
    prioRTVSTSAN_str = ''
    init_latency_str = ''
    
    topics_str = ''
    apps_str = ''
    opt_str = ''
    
    topics = set()
    apps = set()
    subscriptions = set()
    
    with open(dataset_path, 'r') as csvfile:
        dictReader = csv.DictReader(csvfile)
        for row in dictReader:
            topic = row['topic']
            app = row['app']
            topics.add(topic)
            apps.add(app)
            subscriptions.add(topic + "_" + app)
        csvfile.close()
        
    with open(dataset_path, 'r') as csvfile:
        dictReader = csv.DictReader(csvfile)
        for row in dictReader:
            topic = row['topic']
            app = row['app']
            change_baseline = row['baseline']
            change_droppingVS1 = row['dropVS1']
            change_droppingVS2 = row['dropVS2']
            change_droppingVS2AN2 = row['dropVS2AN2']
            change_prioRT = row['prioRT']
            change_prioAN = row['prioAN']
            change_prioTS = row['prioTS']
            change_prioVS = row['prioVS']
            change_prioRTVS = row['prioRTVS']
            change_prioRTVSTSAN = row['prioRTVSTSAN']

            init_latency_str += '\n(= (latency ' + topic + ' ' + app + ') 0)'
            
            if (float(change_baseline) >= 0):
                baseline_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_baseline), 2)) + ')'
            else:
                change_baseline = float(change_baseline)
                change_baseline = change_baseline * -1
                baseline_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_baseline, 2)) + ')'
                
            if (float(change_droppingVS1) >= 0):
                droppingVS1_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS1), 2)) + ')'
            else:
                change_droppingVS1 = float(change_droppingVS1)
                change_droppingVS1 = change_droppingVS1 * -1
                droppingVS1_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS1, 2)) + ')'
                
            if (float(change_droppingVS2) >= 0):
                droppingVS2_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS2), 2)) + ')'
            else:
                change_droppingVS2 = float(change_droppingVS2)
                change_droppingVS2 = change_droppingVS2 * -1
                droppingVS2_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS2, 2)) + ')'
                
            if (float(change_droppingVS2AN2) >= 0):
                droppingVS2AN2_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS2AN2), 2)) + ')'
            else:
                change_droppingVS2AN2 = float(change_droppingVS2AN2)
                change_droppingVS2AN2 = change_droppingVS2AN2 * -1
                droppingVS2AN2_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS2AN2, 2)) + ')'
                
            if (float(change_prioRT) >= 0):
                prioRT_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioRT), 2)) + ')'
            else:
                change_prioRT = float(change_prioRT)
                change_prioRT = change_prioRT * -1
                prioRT_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioRT, 2)) + ')'
                
            if (float(change_prioAN) >= 0):
                prioAN_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioAN), 2)) + ')'
            else:
                change_prioAN = float(change_prioAN)
                change_prioAN = change_prioAN * -1
                prioAN_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioAN, 2)) + ')'
                
            if (float(change_prioTS) >= 0):
                prioTS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioTS), 2)) + ')'
            else:
                change_prioTS = float(change_prioTS)
                change_prioTS = change_prioTS * -1
                prioTS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioTS, 2)) + ')'
                
            if (float(change_prioVS) >= 0):
                prioVS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioVS), 2)) + ')'
            else:
                change_prioVS = float(change_prioVS)
                change_prioVS = change_prioVS * -1
                prioVS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioVS, 2)) + ')'
                
            if (float(change_prioRTVS) >= 0):
                prioRTVS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioRTVS), 2)) + ')'
            else:
                change_prioRTVS = float(change_prioRTVS)
                change_prioRTVS = change_prioRTVS * -1
                prioRTVS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioRTVS, 2)) + ')'
                
            if (float(change_prioRTVSTSAN) >= 0):
                prioRTVSTSAN_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioRTVSTSAN), 2)) + ')'
            else:
                change_prioRTVSTSAN = float(change_prioRTVSTSAN)
                change_prioRTVSTSAN = change_prioRTVSTSAN * -1
                prioRTVSTSAN_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioRTVSTSAN, 2)) + ')'
                
        csvfile.close()
        
        for topic in topics:
            topics_str += topic + ' '
        for app in apps:
            apps_str += app + ' '
            
        with open(domain_template, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('#baseline_effects#', baseline_str)
            filedata = filedata.replace('#dropVS1_effects#', droppingVS1_str)
            filedata = filedata.replace('#dropVS2_effects#', droppingVS2_str)
            filedata = filedata.replace('#dropVS2AN2_effects#', droppingVS2AN2_str)
            filedata = filedata.replace('#prioRT_effects#', prioRT_str)
            filedata = filedata.replace('#prioAN_effects#', prioAN_str)
            filedata = filedata.replace('#prioTS_effects#', prioTS_str)
            filedata = filedata.replace('#prioVS_effects#', prioVS_str)
            filedata = filedata.replace('#prioRTVS_effects#', prioRTVS_str)
            filedata = filedata.replace('#prioRTVSTSAN_effects#', prioRTVSTSAN_str)
                
            with open(domain_generated, 'w') as file:
                  file.write(filedata)
                  file.close()
            file.close()

        opt_str = '(+ ' * ( len(subscriptions) - 1)
        counter = 1
        for subscription in subscriptions:
            parts = subscription.split('_')
            topic = parts[0]
            app = parts[1]
            if counter == 1:
                opt_str += '(* 1 (latency ' + topic + ' ' + app + ' )) \n'
                counter += 1
            else:
                opt_str += '(* 1 (latency ' + topic + ' ' + app + ' )))\n'
        opt_str += ')'
        
        
        with open(problem_template, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('#topics#', topics_str)
            filedata = filedata.replace('#apps#', apps_str)
            filedata = filedata.replace('#init_predicates#', init_latency_str)
            filedata = filedata.replace('#metric#', opt_str)
            with open(problem_generated, 'w') as file:
                  file.write(filedata)

def instantiate_files_overloaded(dataset_path, domain_template, problem_template, domain_generated, problem_generated):
    baseline_str = ''
    droppingAN10_str = ''
    droppingRT10_str = ''
    droppingVS10_str = ''
    droppingVS10AN10_str = ''
    droppingVS10AN10RT10_str = ''
    droppingVS15AN15_str = ''

    prioRT_str = ''
    prioAN_str = ''
    prioTS_str = ''
    prioVS_str = ''
    prioRTVS_str = ''
    prioRTVSTSAN_str = ''
    init_latency_str = ''
    
    topics_str = ''
    apps_str = ''
    opt_str = ''
    
    topics = set()
    apps = set()
    subscriptions = set()
    
    with open(dataset_path, 'r') as csvfile:
        dictReader = csv.DictReader(csvfile)
        for row in dictReader:
            topic = row['topic']
            app = row['app']
            topics.add(topic)
            apps.add(app)
            subscriptions.add(topic + "_" + app)
        csvfile.close()
        
    with open(dataset_path, 'r') as csvfile:
        dictReader = csv.DictReader(csvfile)
        for row in dictReader:
            topic = row['topic']
            app = row['app']
            change_baseline = row['baseline']

            change_droppingAN10 = row['dropAN10']
            change_droppingRT10 = row['dropRT10']
            change_droppingVS10 = row['dropVS10']
            change_droppingVS10AN10 = row['dropVS10AN10']
            change_droppingVS10AN10RT10 = row['dropVS10AN10RT10']
            change_droppingVS15AN15 = row['dropVS15AN15']
            
            change_prioRT = row['prioRT']
            change_prioAN = row['prioAN']
            change_prioTS = row['prioTS']
            change_prioVS = row['prioVS']
            change_prioRTVS = row['prioRTVS']
            change_prioRTVSTSAN = row['prioRTVSTSAN']
            
            init_latency_str += '\n(= (latency ' + topic + ' ' + app + ') 0)'
            
            if (float(change_baseline) >= 0):
                baseline_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_baseline), 2)) + ')'
            else:
                change_baseline = float(change_baseline)
                change_baseline = change_baseline * -1
                baseline_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_baseline, 2)) + ')'
                
            if (float(change_droppingAN10) >= 0):
                droppingAN10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingAN10), 2)) + ')'
            else:
                change_droppingAN10 = float(change_droppingAN10)
                change_droppingAN10 = change_droppingAN10 * -1
                droppingAN10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingAN10, 2)) + ')'    
                
            if (float(change_droppingRT10) >= 0):
                droppingRT10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingRT10), 2)) + ')'
            else:
                change_droppingRT10 = float(change_droppingRT10)
                change_droppingRT10 = change_droppingRT10 * -1
                droppingRT10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingRT10, 2)) + ')'     
            
            if (float(change_droppingVS10) >= 0):
                droppingVS10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS10), 2)) + ')'
            else:
                change_droppingVS10 = float(change_droppingVS10)
                change_droppingVS10 = change_droppingVS10 * -1
                droppingVS10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS10, 2)) + ')'  
                
            if (float(change_droppingVS10AN10) >= 0):
                droppingVS10AN10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS10AN10), 2)) + ')'
            else:
                change_droppingVS10AN10 = float(change_droppingVS10AN10)
                change_droppingVS10AN10 = change_droppingVS10AN10 * -1
                droppingVS10AN10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS10AN10, 2)) + ')' 
    
            if (float(change_droppingVS10AN10RT10) >= 0):
                droppingVS10AN10RT10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS10AN10RT10), 2)) + ')'
            else:
                change_droppingVS10AN10RT10 = float(change_droppingVS10AN10RT10)
                change_droppingVS10AN10RT10 = change_droppingVS10AN10RT10 * -1
                droppingVS10AN10RT10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS10AN10RT10, 2)) + ')' 
                
            if (float(change_droppingVS15AN15) >= 0):
                droppingVS15AN15_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS15AN15), 2)) + ')'
            else:
                change_droppingVS15AN15 = float(change_droppingVS15AN15)
                change_droppingVS15AN15 = change_droppingVS15AN15 * -1
                droppingVS15AN15_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS15AN15, 2)) + ')' 
                
            if (float(change_prioRT) >= 0):
                prioRT_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioRT), 2)) + ')'
            else:
                change_prioRT = float(change_prioRT)
                change_prioRT = change_prioRT * -1
                prioRT_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioRT, 2)) + ')'
                
            if (float(change_prioAN) >= 0):
                prioAN_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioAN), 2)) + ')'
            else:
                change_prioAN = float(change_prioAN)
                change_prioAN = change_prioAN * -1
                prioAN_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioAN, 2)) + ')'
                
            if (float(change_prioTS) >= 0):
                prioTS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioTS), 2)) + ')'
            else:
                change_prioTS = float(change_prioTS)
                change_prioTS = change_prioTS * -1
                prioTS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioTS, 2)) + ')'
                
            if (float(change_prioVS) >= 0):
                prioVS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioVS), 2)) + ')'
            else:
                change_prioVS = float(change_prioVS)
                change_prioVS = change_prioVS * -1
                prioVS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioVS, 2)) + ')'
                
            if (float(change_prioRTVS) >= 0):
                prioRTVS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioRTVS), 2)) + ')'
            else:
                change_prioRTVS = float(change_prioRTVS)
                change_prioRTVS = change_prioRTVS * -1
                prioRTVS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioRTVS, 2)) + ')'
                
            if (float(change_prioRTVSTSAN) >= 0):
                prioRTVSTSAN_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioRTVSTSAN), 2)) + ')'
            else:
                change_prioRTVSTSAN = float(change_prioRTVSTSAN)
                change_prioRTVSTSAN = change_prioRTVSTSAN * -1
                prioRTVSTSAN_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioRTVSTSAN, 2)) + ')'
            
        csvfile.close()
        
        for topic in topics:
            topics_str += topic + ' '
        for app in apps:
            apps_str += app + ' '
            
        with open(domain_template, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('#baseline_effects#', baseline_str)

            filedata = filedata.replace('#dropAN10_effects#', droppingAN10_str)
            filedata = filedata.replace('#dropVS10_effects#', droppingVS10_str)
            filedata = filedata.replace('#dropRT10_effects#', droppingRT10_str)
            filedata = filedata.replace('#dropVS10AN10_effects#', droppingVS10AN10_str)
            filedata = filedata.replace('#dropVS10AN10RT10_effects#', droppingVS10AN10RT10_str)
            filedata = filedata.replace('#dropVS15AN15_effects#', droppingVS15AN15_str)
            
            filedata = filedata.replace('#prioRT_effects#', prioRT_str)
            filedata = filedata.replace('#prioAN_effects#', prioAN_str)
            filedata = filedata.replace('#prioTS_effects#', prioTS_str)
            filedata = filedata.replace('#prioVS_effects#', prioVS_str)
            filedata = filedata.replace('#prioRTVS_effects#', prioRTVS_str)
            filedata = filedata.replace('#prioRTVSTSAN_effects#', prioRTVSTSAN_str)
                
            with open(domain_generated, 'w') as file:
                  file.write(filedata)
                  file.close()

        opt_str = '(+ ' * ( len(subscriptions) - 1)
        counter = 1
        for subscription in subscriptions:
            parts = subscription.split('_')
            topic = parts[0]
            app = parts[1]
            if counter == 1:
                opt_str += '(* 1 (latency ' + topic + ' ' + app + ' )) \n'
                counter += 1
            else:
                opt_str += '(* 1 (latency ' + topic + ' ' + app + ' )))\n'
        opt_str += ')'
        
        with open(problem_template, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('#topics#', topics_str)
            filedata = filedata.replace('#apps#', apps_str)
            filedata = filedata.replace('#init_predicates#', init_latency_str)
            filedata = filedata.replace('#metric#', opt_str)
            with open(problem_generated, 'w') as file:
                  file.write(filedata)
                  file.close()
            file.close()
        file.close()
 

def instantiate_files_emergency(dataset_path, domain_template, problem_template, domain_generated, problem_generated):
    baseline_str = ''
    
    droppingVS15AN15_str = ''
    droppingVS15AN15RT15_str = ''
    droppingVS20AN20_str = ''
    droppingVS20AN20RT10_str = ''

    prioEM_str = ''
    prioEM0RT1_str = ''
    prioEMRT_str = ''
    prioEMRTVS_str = ''
    prioEMRTVSTSAN_str = ''

    init_latency_str = ''
    
    topics_str = ''
    apps_str = ''
    opt_str = ''
    
    topics = set()
    apps = set()
    subscriptions = set()
    
    with open(dataset_path, 'r') as csvfile:
        dictReader = csv.DictReader(csvfile)
        for row in dictReader:
            topic = row['topic']
            app = row['app']
            topics.add(topic)
            apps.add(app)
            subscriptions.add(topic + "_" + app)
        csvfile.close()
        
    with open(dataset_path, 'r') as csvfile:
        dictReader = csv.DictReader(csvfile)
        for row in dictReader:
            topic = row['topic']
            app = row['app']
            change_baseline = row['baseline']

            change_droppingVS15AN15 = row['dropVS15AN15']
            change_droppingVS15AN15RT15 = row['dropVS15AN15RT15']
            change_droppingVS20AN20 = row['dropVS20AN20']
            change_droppingVS20AN20RT10 = row['dropVS20AN20RT10']
            
            change_prioEM = row['prioEM']
            change_prioEM0RT1 = row['prioEM0RT1']
            change_prioEMRT = row['prioEMRT']
            change_prioEMRTVS = row['prioEMRTVS']
            change_prioEMRTVSTSAN = row['prioEMRTVSTSAN']
            
            init_latency_str += '\n(= (latency ' + topic + ' ' + app + ') 0)'
            
            if (float(change_baseline) >= 0):
                baseline_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_baseline), 2)) + ')'
            else:
                change_baseline = float(change_baseline)
                change_baseline = change_baseline * -1
                baseline_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_baseline, 2)) + ')'
                
            if (float(change_droppingVS15AN15) >= 0):
                droppingVS15AN15_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS15AN15), 2)) + ')'
            else:
                change_droppingVS15AN15 = float(change_droppingVS15AN15)
                change_droppingVS15AN15 = change_droppingVS15AN15 * -1
                droppingVS15AN15_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS15AN15, 2)) + ')'    
                
            if (float(change_droppingVS15AN15RT15) >= 0):
                droppingVS15AN15RT15_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS15AN15RT15), 2)) + ')'
            else:
                change_droppingVS15AN15RT15 = float(change_droppingVS15AN15RT15)
                change_droppingVS15AN15RT15 = change_droppingVS15AN15RT15 * -1
                droppingVS15AN15RT15_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS15AN15RT15, 2)) + ')'     
            
            if (float(change_droppingVS20AN20) >= 0):
                droppingVS20AN20_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS20AN20), 2)) + ')'
            else:
                change_droppingVS20AN20 = float(change_droppingVS20AN20)
                change_droppingVS20AN20 = change_droppingVS20AN20 * -1
                droppingVS20AN20_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS20AN20, 2)) + ')'  
                
            if (float(change_droppingVS20AN20RT10) >= 0):
                droppingVS20AN20RT10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_droppingVS20AN20RT10), 2)) + ')'
            else:
                change_droppingVS20AN20RT10 = float(change_droppingVS20AN20RT10)
                change_droppingVS20AN20RT10 = change_droppingVS20AN20RT10 * -1
                droppingVS20AN20RT10_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_droppingVS20AN20RT10, 2)) + ')' 
                
            if (float(change_prioEM) >= 0):
                prioEM_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioEM), 2)) + ')'
            else:
                change_prioEM = float(change_prioEM)
                change_prioEM = change_prioEM * -1
                prioEM_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioEM, 2)) + ')'
                
            if (float(change_prioEM0RT1) >= 0):
                prioEM0RT1_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioEM0RT1), 2)) + ')'
            else:
                change_prioEM0RT1 = float(change_prioEM0RT1)
                change_prioEM0RT1 = change_prioEM0RT1 * -1
                prioEM0RT1_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioEM0RT1, 2)) + ')'
                
            if (float(change_prioEMRT) >= 0):
                prioEMRT_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioEMRT), 2)) + ')'
            else:
                change_prioEMRT = float(change_prioEMRT)
                change_prioEMRT = change_prioEMRT * -1
                prioEMRT_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioEMRT, 2)) + ')'
                
            if (float(change_prioEMRTVS) >= 0):
                prioEMRTVS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioEMRTVS), 2)) + ')'
            else:
                change_prioEMRTVS = float(change_prioEMRTVS)
                change_prioEMRTVS = change_prioEMRTVS * -1
                prioEMRTVS_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioEMRTVS, 2)) + ')'
                
            if (float(change_prioEMRTVSTSAN) >= 0):
                prioEMRTVSTSAN_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(float(change_prioEMRTVSTSAN), 2)) + ')'
            else:
                change_prioEMRTVSTSAN = float(change_prioEMRTVSTSAN)
                change_prioEMRTVSTSAN = change_prioEMRTVSTSAN * -1
                prioEMRTVSTSAN_str += '\n(increase (latency ' + topic + ' ' + app + ') ' + str(round(change_prioEMRTVSTSAN, 2)) + ')'
            
        csvfile.close()
        
        for topic in topics:
            topics_str += topic + ' '
        for app in apps:
            apps_str += app + ' '
            
        with open(domain_template, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('#baseline_effects#', baseline_str)

            filedata = filedata.replace('#dropVS15AN15_effects#', droppingVS15AN15_str)
            filedata = filedata.replace('#dropVS15AN15RT15_effects#', droppingVS15AN15RT15_str)
            filedata = filedata.replace('#dropVS20AN20_effects#', droppingVS20AN20_str)
            filedata = filedata.replace('#dropVS20AN20RT10_effects#', droppingVS20AN20RT10_str)
            
            filedata = filedata.replace('#prioEM_effects#', prioEM_str)
            filedata = filedata.replace('#prioEMRT_effects#', prioEMRT_str)
            filedata = filedata.replace('#prioEM0RT1_effects#', prioEM0RT1_str)
            filedata = filedata.replace('#prioEMRTVS_effects#', prioEMRTVS_str)
            filedata = filedata.replace('#prioEMRTVSTSAN_effects#', prioEMRTVSTSAN_str)
                
            with open(domain_generated, 'w') as file:
                  file.write(filedata)
                  file.close()

        opt_str = '(+ ' * ( len(subscriptions) - 1)
        counter = 1
        for subscription in subscriptions:
            parts = subscription.split('_')
            topic = parts[0]
            app = parts[1]
            if counter == 1:
                opt_str += '(* 1 (latency ' + topic + ' ' + app + ' )) \n'
                counter += 1
            else:
                opt_str += '(* 1 (latency ' + topic + ' ' + app + ' )))\n'
        opt_str += ')'
        
        with open(problem_template, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('#topics#', topics_str)
            filedata = filedata.replace('#apps#', apps_str)
            filedata = filedata.replace('#init_predicates#', init_latency_str)
            filedata = filedata.replace('#metric#', opt_str)
            with open(problem_generated, 'w') as file:
                  file.write(filedata)
                  file.close()
            file.close()
        file.close()             

def main():
    args = sys.argv[1:]
    if (len(args) < 3 or len(args) > 4):
        print('Wrong number of arguments.')
        print('Usage: python InstantiatePddlTemplates.py [dataset_path] [domain_template] [problem_template] [option]')
        return
    dataset_path = args[0]
    domain_template_path = args[1]
    problem_template_path = args[2]
    flag = ''
    if (len(args) == 4):
        flag = args[3]
        
    if (not os.path.isfile(dataset_path)):
        print('Dataset does not exist or is not a file.')
        return
    if (not os.path.isfile(domain_template_path)):
        print('Domain template does not exist or is not a file.')
        return
    if (not os.path.isfile(problem_template_path)):
        print('Problem template does not exist or is not a file.')
        return
        
    domain_generated_path = get_path_of_generated(domain_template_path)
    problem_generated_path = get_path_of_generated(problem_template_path)
    if (flag == '-o'):
        instantiate_files_overloaded(dataset_path, domain_template_path, problem_template_path, 
                                     domain_generated_path, problem_generated_path)
    elif (flag == '-e'):
        instantiate_files_emergency(dataset_path, domain_template_path, problem_template_path, 
                                     domain_generated_path, problem_generated_path)
    else:
        instantiate_files(dataset_path, domain_template_path, problem_template_path, 
                                     domain_generated_path, problem_generated_path)
    
if __name__ == "__main__":
    main()