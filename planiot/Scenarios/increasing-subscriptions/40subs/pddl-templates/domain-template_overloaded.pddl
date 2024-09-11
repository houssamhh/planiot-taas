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

(:action droppingVS10
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
       #dropVS10_effects#
    )
)

(:action droppingAN10
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        #dropAN10_effects#
    )
)


(:action droppingRT10
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        #dropRT10_effects#
    )
)

(:action droppingVS10AN10
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        #dropVS10AN10_effects#
    )
)

(:action droppingVS10AN10RT10
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        #dropVS10AN10RT10_effects#
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



(:action prioritize_RT
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #prioRT_effects#
    )
)

(:action prioritize_AN
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #prioAN_effects#
    )
)

(:action prioritize_TS
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #prioTS_effects#
    )
)

(:action prioritize_VS
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #prioVS_effects#
    )
)


(:action prioritize_RT_VS_TS_AN
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        #prioRTVSTSAN_effects#
    )
)

)
