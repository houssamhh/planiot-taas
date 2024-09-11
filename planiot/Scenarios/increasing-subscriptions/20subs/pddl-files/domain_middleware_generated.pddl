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
        
(increase (latency intrusiondetection app2) 0.21)
(increase (latency intrusiondetection app21) 0.21)
(increase (latency videosurveillance app38) 0.21)
(increase (latency intrusiondetection app36) 0.21)
(increase (latency smartthings app1) 0.18)
(increase (latency bms app2) 0.17)
(increase (latency occupancymanagement app21) 0.17)
(increase (latency printing app2) 0.17)
(increase (latency firedetection app23) 0.21)
(increase (latency firedetection app22) 0.21)
(increase (latency amazonecho app21) 0.13)
(increase (latency printing app11) 0.17)
(increase (latency energymanagement app2) 0.17)
(increase (latency energymanagement app1) 0.17)
(increase (latency smartthings app22) 0.18)
(increase (latency firedetection app1) 0.21)
(increase (latency bms app21) 0.17)
(increase (latency occupancymanagement app12) 0.17)
(increase (latency videosurveillance app2) 0.21)
(increase (latency videosurveillance app36) 0.21)
    )
)

(:action droppingVS1
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
       
(increase (latency intrusiondetection app2) 0.21)
(increase (latency intrusiondetection app21) 0.21)
(increase (latency videosurveillance app38) 0.21)
(increase (latency intrusiondetection app36) 0.21)
(increase (latency smartthings app1) 0.17)
(increase (latency bms app2) 0.17)
(increase (latency occupancymanagement app21) 0.18)
(increase (latency printing app2) 0.16)
(increase (latency firedetection app23) 0.21)
(increase (latency firedetection app22) 0.2)
(increase (latency amazonecho app21) 0.13)
(increase (latency printing app11) 0.17)
(increase (latency energymanagement app2) 0.17)
(increase (latency energymanagement app1) 0.17)
(increase (latency smartthings app22) 0.16)
(increase (latency firedetection app1) 0.2)
(increase (latency bms app21) 0.17)
(increase (latency occupancymanagement app12) 0.17)
(increase (latency videosurveillance app2) 0.21)
(increase (latency videosurveillance app36) 0.21)
    )
)

(:action droppingVS2
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        
(increase (latency intrusiondetection app2) 0.21)
(increase (latency intrusiondetection app21) 0.21)
(increase (latency videosurveillance app38) 0.21)
(increase (latency intrusiondetection app36) 0.21)
(increase (latency smartthings app1) 0.17)
(increase (latency bms app2) 0.17)
(increase (latency occupancymanagement app21) 0.17)
(increase (latency printing app2) 0.17)
(increase (latency firedetection app23) 0.21)
(increase (latency firedetection app22) 0.2)
(increase (latency amazonecho app21) 0.12)
(increase (latency printing app11) 0.17)
(increase (latency energymanagement app2) 0.17)
(increase (latency energymanagement app1) 0.17)
(increase (latency smartthings app22) 0.17)
(increase (latency firedetection app1) 0.22)
(increase (latency bms app21) 0.17)
(increase (latency occupancymanagement app12) 0.17)
(increase (latency videosurveillance app2) 0.21)
(increase (latency videosurveillance app36) 0.2)
    )
)


(:action droppingVS2AN2
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        
(increase (latency intrusiondetection app2) 0.21)
(increase (latency intrusiondetection app21) 0.2)
(increase (latency videosurveillance app38) 0.2)
(increase (latency intrusiondetection app36) 0.2)
(increase (latency smartthings app1) 0.17)
(increase (latency bms app2) 0.17)
(increase (latency occupancymanagement app21) 0.16)
(increase (latency printing app2) 0.17)
(increase (latency firedetection app23) 0.21)
(increase (latency firedetection app22) 0.2)
(increase (latency amazonecho app21) 0.13)
(increase (latency printing app11) 0.17)
(increase (latency energymanagement app2) 0.16)
(increase (latency energymanagement app1) 0.16)
(increase (latency smartthings app22) 0.17)
(increase (latency firedetection app1) 0.21)
(increase (latency bms app21) 0.17)
(increase (latency occupancymanagement app12) 0.17)
(increase (latency videosurveillance app2) 0.21)
(increase (latency videosurveillance app36) 0.21)
    )
)

(:action droppingVS2AN5
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (baseline ?t  ?app )  
    )
    :effect (and 
        (not (baseline ?t  ?app ))
        (QoS_achieved ?t  ?app )
        
(increase (latency intrusiondetection app2) 0.2)
(increase (latency intrusiondetection app21) 0.2)
(increase (latency videosurveillance app38) 0.2)
(increase (latency intrusiondetection app36) 0.2)
(increase (latency smartthings app1) 0.17)
(increase (latency bms app2) 0.17)
(increase (latency occupancymanagement app21) 0.17)
(increase (latency printing app2) 0.17)
(increase (latency firedetection app23) 0.19)
(increase (latency firedetection app22) 0.19)
(increase (latency amazonecho app21) 0.13)
(increase (latency printing app11) 0.16)
(increase (latency energymanagement app2) 0.16)
(increase (latency energymanagement app1) 0.16)
(increase (latency smartthings app22) 0.17)
(increase (latency firedetection app1) 0.2)
(increase (latency bms app21) 0.17)
(increase (latency occupancymanagement app12) 0.17)
(increase (latency videosurveillance app2) 0.2)
(increase (latency videosurveillance app36) 0.2)
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
        
(increase (latency intrusiondetection app2) 0.21)
(increase (latency intrusiondetection app21) 0.21)
(increase (latency videosurveillance app38) 0.21)
(increase (latency intrusiondetection app36) 0.21)
(increase (latency smartthings app1) 0.18)
(increase (latency bms app2) 0.17)
(increase (latency occupancymanagement app21) 0.17)
(increase (latency printing app2) 0.17)
(increase (latency firedetection app23) 0.21)
(increase (latency firedetection app22) 0.21)
(increase (latency amazonecho app21) 0.13)
(increase (latency printing app11) 0.17)
(increase (latency energymanagement app2) 0.17)
(increase (latency energymanagement app1) 0.17)
(increase (latency smartthings app22) 0.18)
(increase (latency firedetection app1) 0.21)
(increase (latency bms app21) 0.17)
(increase (latency occupancymanagement app12) 0.17)
(increase (latency videosurveillance app2) 0.21)
(increase (latency videosurveillance app36) 0.21)
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
        
(increase (latency intrusiondetection app2) 0.24)
(increase (latency intrusiondetection app21) 0.13)
(increase (latency videosurveillance app38) 0.22)
(increase (latency intrusiondetection app36) 0.25)
(increase (latency smartthings app1) 0.19)
(increase (latency bms app2) 0.19)
(increase (latency occupancymanagement app21) 0.12)
(increase (latency printing app2) 0.17)
(increase (latency firedetection app23) 0.15)
(increase (latency firedetection app22) 0.15)
(increase (latency amazonecho app21) 0.1)
(increase (latency printing app11) 0.18)
(increase (latency energymanagement app2) 0.18)
(increase (latency energymanagement app1) 0.18)
(increase (latency smartthings app22) 0.12)
(increase (latency firedetection app1) 0.26)
(increase (latency bms app21) 0.12)
(increase (latency occupancymanagement app12) 0.19)
(increase (latency videosurveillance app2) 0.22)
(increase (latency videosurveillance app36) 0.22)
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
        
(increase (latency intrusiondetection app2) 0.13)
(increase (latency intrusiondetection app21) 0.25)
(increase (latency videosurveillance app38) 0.25)
(increase (latency intrusiondetection app36) 0.25)
(increase (latency smartthings app1) 0.13)
(increase (latency bms app2) 0.12)
(increase (latency occupancymanagement app21) 0.17)
(increase (latency printing app2) 0.12)
(increase (latency firedetection app23) 0.25)
(increase (latency firedetection app22) 0.25)
(increase (latency amazonecho app21) 0.14)
(increase (latency printing app11) 0.19)
(increase (latency energymanagement app2) 0.14)
(increase (latency energymanagement app1) 0.14)
(increase (latency smartthings app22) 0.2)
(increase (latency firedetection app1) 0.13)
(increase (latency bms app21) 0.2)
(increase (latency occupancymanagement app12) 0.17)
(increase (latency videosurveillance app2) 0.13)
(increase (latency videosurveillance app36) 0.25)
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
        
(increase (latency intrusiondetection app2) 0.21)
(increase (latency intrusiondetection app21) 0.21)
(increase (latency videosurveillance app38) 0.21)
(increase (latency intrusiondetection app36) 0.21)
(increase (latency smartthings app1) 0.18)
(increase (latency bms app2) 0.17)
(increase (latency occupancymanagement app21) 0.19)
(increase (latency printing app2) 0.19)
(increase (latency firedetection app23) 0.21)
(increase (latency firedetection app22) 0.21)
(increase (latency amazonecho app21) 0.15)
(increase (latency printing app11) 0.12)
(increase (latency energymanagement app2) 0.17)
(increase (latency energymanagement app1) 0.17)
(increase (latency smartthings app22) 0.17)
(increase (latency firedetection app1) 0.22)
(increase (latency bms app21) 0.17)
(increase (latency occupancymanagement app12) 0.12)
(increase (latency videosurveillance app2) 0.21)
(increase (latency videosurveillance app36) 0.21)
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
        
(increase (latency intrusiondetection app2) 0.25)
(increase (latency intrusiondetection app21) 0.25)
(increase (latency videosurveillance app38) 0.15)
(increase (latency intrusiondetection app36) 0.13)
(increase (latency smartthings app1) 0.18)
(increase (latency bms app2) 0.18)
(increase (latency occupancymanagement app21) 0.18)
(increase (latency printing app2) 0.18)
(increase (latency firedetection app23) 0.21)
(increase (latency firedetection app22) 0.21)
(increase (latency amazonecho app21) 0.14)
(increase (latency printing app11) 0.18)
(increase (latency energymanagement app2) 0.18)
(increase (latency energymanagement app1) 0.18)
(increase (latency smartthings app22) 0.18)
(increase (latency firedetection app1) 0.22)
(increase (latency bms app21) 0.18)
(increase (latency occupancymanagement app12) 0.18)
(increase (latency videosurveillance app2) 0.27)
(increase (latency videosurveillance app36) 0.15)
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
        
(increase (latency intrusiondetection app2) 0.3)
(increase (latency intrusiondetection app21) 0.13)
(increase (latency videosurveillance app38) 0.17)
(increase (latency intrusiondetection app36) 0.19)
(increase (latency smartthings app1) 0.22)
(increase (latency bms app2) 0.21)
(increase (latency occupancymanagement app21) 0.12)
(increase (latency printing app2) 0.21)
(increase (latency firedetection app23) 0.15)
(increase (latency firedetection app22) 0.15)
(increase (latency amazonecho app21) 0.09)
(increase (latency printing app11) 0.15)
(increase (latency energymanagement app2) 0.2)
(increase (latency energymanagement app1) 0.19)
(increase (latency smartthings app22) 0.12)
(increase (latency firedetection app1) 0.3)
(increase (latency bms app21) 0.12)
(increase (latency occupancymanagement app12) 0.18)
(increase (latency videosurveillance app2) 0.29)
(increase (latency videosurveillance app36) 0.17)
    )
)

(:action prioritize_RT_VS
    :parameters (?t  - topic ?app - application )
    :precondition (and 
            (priority_not_set ?t  ?app )  
    )
    :effect (and 
        (not (priority_not_set ?t  ?app ))
        (priority_set ?t  ?app )
        
(increase (latency intrusiondetection app2) 0.3)
(increase (latency intrusiondetection app21) 0.13)
(increase (latency videosurveillance app38) 0.17)
(increase (latency intrusiondetection app36) 0.19)
(increase (latency smartthings app1) 0.22)
(increase (latency bms app2) 0.21)
(increase (latency occupancymanagement app21) 0.12)
(increase (latency printing app2) 0.19)
(increase (latency firedetection app23) 0.15)
(increase (latency firedetection app22) 0.15)
(increase (latency amazonecho app21) 0.1)
(increase (latency printing app11) 0.19)
(increase (latency energymanagement app2) 0.19)
(increase (latency energymanagement app1) 0.19)
(increase (latency smartthings app22) 0.12)
(increase (latency firedetection app1) 0.29)
(increase (latency bms app21) 0.12)
(increase (latency occupancymanagement app12) 0.21)
(increase (latency videosurveillance app2) 0.3)
(increase (latency videosurveillance app36) 0.17)
    )
)

)
