(define (problem problem_name) (:domain domain_name)
(:objects 
    #topics# topic_all  - Topic
    #apps# app_all - Application
    ;;dropping10 dropping20 baseline priorities finiteCapacity  - QosConfig
    ;;shared-bw maxmin-bw topics-bw  - NetworkPolicy
)
(:init
    ;todo: put the initial state's facts and numeric values here
        (baseline topic_all app_all)
        (priority_not_set topic_all app_all)
        
    #init_predicates#
)
(:goal (and
    ;todo: put the goal condition here
        (QoS_achieved topic_all app_all)
        (priority_set topic_all app_all)
))
;un-comment the following line if metric is needed
(:metric minimize #metric#
)
