;Header and description

(define (domain domain_name)

;remove requirements that are not needed
(:requirements :strips :fluents :typing :conditional-effects :negative-preconditions :equality :disjunctive-preconditions)

(:types ;todo: enumerate types and their hierarchy here, e.g. car truck bus - vehicle
    Topic - object                  ;;because in JMT we will get results for each topic
    Application - object            ;;an app that subscribes to a topic
)

; un-comment following line if constants are needed
(:constants  true false)

(:predicates ;todo: define predicates here

    (baseline ?t - Topic ?app - Application)  
    (QoS_achieved ?t - Topic ?app - Application)
    (priority_not_set ?t - Topic ?app - Application)  
    (priority_set ?t - Topic ?app - Application)
    ;;bandwidth allocation policy
    ;;(shared ?policy - NetworkPolicy)
    ;;(topics ?policy - NetworkPolicy)
    ;;(maxmin ?policy - NetworkPolicy)
)


(:functions ;todo: define numeric functions here
    (latency ?topic ?app) - number
)

;define actions here


(:action no_change
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        #baseline_effects#
    )
)


(:action droppingVS15AN15
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        #dropVS15AN15_effects#
    )
)

(:action droppingVS15AN15RT15
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        #dropVS15AN15RT15_effects#
    )
)

(:action droppingVS20AN20
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        #dropVS20AN20_effects#
    )
)

(:action droppingVS20AN20RT10
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        #dropVS20AN20RT10_effects#
    )
)

(:action no_priority
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #baseline_effects#
    )
)



(:action prioritize_EM
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #prioEM_effects#
    )
)

(:action prioritize_EM_RT
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #prioEMRT_effects#
    )
)

(:action prioritize_EM0_RT1
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #prioEM0RT1_effects#
    )
)

(:action prioritize_EM_RT_VS
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #prioEMRTVS_effects#
    )
)


(:action prioritize_EM_RT_VS_TS_AN
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #prioEMRTVSTSAN_effects#
    )
)

)
