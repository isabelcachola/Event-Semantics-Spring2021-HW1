

# AGENT := ((volition > 0) OR (instigation > 0) AND  (existed-before > 0))
AGENT_QUERY_STR = """
    SELECT DISTINCT ?edge
           WHERE { ?node ?edge ?arg ;
                         <domain> <semantics> ;
                         <type>   <predicate> ;
                         <pred-particular> ?predparticular
                         FILTER ( ?predparticular > 0 ) .
                    ?arg  <domain> <semantics> ;
                         <type>   <argument>  ;
                         <arg-particular> ?argparticular
                         FILTER ( ?argparticular > 0 ) .
                    ?edge <existed_before> ?existed_before
                                FILTER ( ?existed_before > 0 )
                    { ?edge <instigation> ?instigation
                                FILTER ( ?instigation > 0 )
                    } UNION
                    { ?edge <volitation> ?volitation
                                FILTER ( ?volitation > 0 )
                    }
                 }
"""

# PATIENT := ((volition < 0) OR (instigation < 0) AND  (existed-before > 0))
PATIENT_QUERY_STR = """
    SELECT DISTINCT ?edge
           WHERE { ?node ?edge ?arg ;
                         <domain> <semantics> ;
                         <type>   <predicate> ;
                         <pred-particular> ?predparticular
                         FILTER ( ?predparticular > 0 ) .
                    ?arg  <domain> <semantics> ;
                         <type>   <argument>  ;
                         <arg-particular> ?argparticular
                         FILTER ( ?argparticular > 0 ) .
                    ?edge <existed_before> ?existed_before
                                FILTER ( ?existed_before > 0 )
                    { ?edge <instigation> ?instigation
                                FILTER ( ?instigation < 0 )
                    } UNION
                    { ?edge <volitation> ?volitation
                                FILTER ( ?volitation < 0 )
                    }
                 }
"""

# THEME := (change_of_location > 0) AND  (volition < 0) AND (existed_before > 0)
THEME_QUERY_STR = """
    SELECT DISTINCT ?edge
           WHERE { ?node ?edge ?arg ;
                         <domain> <semantics> ;
                         <type>   <predicate> ;
                         <pred-particular> ?predparticular
                         FILTER ( ?predparticular > 0 ) .
                    ?arg  <domain> <semantics> ;
                         <type>   <argument>  ;
                         <arg-particular> ?argparticular
                         FILTER ( ?argparticular > 0 ) .
                    ?edge <existed_before> ?existed_before
                                FILTER ( ?existed_before > 0 )
                    ?edge <change_of_location> ?change_of_location
                                FILTER ( ?change_of_location > 0 )
                    ?edge <volition> ?volition
                                FILTER ( ?volition < 0 )
                    
                 }
"""

# EXPERIENCER := ((change_of_state_continuous > 0) AND (volition < 0) AND (awareness > 0)
EXPERIENCER_QUERY_STR = """
    SELECT DISTINCT ?edge
           WHERE { ?node ?edge ?arg ;
                         <domain> <semantics> ;
                         <type>   <predicate> ;
                         <pred-particular> ?predparticular
                         FILTER ( ?predparticular > 0 ) .
                    ?arg  <domain> <semantics> ;
                         <type>   <argument>  ;
                         <arg-particular> ?argparticular
                         FILTER ( ?argparticular > 0 ) .
                    ?edge <volition> ?volition
                                FILTER ( ?volition < 0 )
                    ?edge <awareness> ?awareness
                                FILTER ( ?awareness > 0 )
                    ?edge <change_of_state_continuous> ?change_of_state_continuous
                                FILTER ( ?change_of_state_continuous > 0 )
                 }
"""

# RECIPIENT := (change_of_possession > 0) AND (existed-before > 0) AND (volition < 0)
RECIPIENT_QUERY_STR = """
    SELECT DISTINCT ?edge
           WHERE { ?node ?edge ?arg ;
                         <domain> <semantics> ;
                         <type>   <predicate> ;
                         <pred-particular> ?predparticular
                         FILTER ( ?predparticular > 0 ) .
                    ?arg  <domain> <semantics> ;
                         <type>   <argument>  ;
                         <arg-particular> ?argparticular
                         FILTER ( ?argparticular > 0 ) .
                    ?edge <existed_before> ?existed_before
                                FILTER ( ?existed_before > 0 )
                    ?edge <change_of_possession> ?change_of_possession
                                FILTER ( ?change_of_possession > 0 )
                    ?edge <volition> ?volition
                                FILTER ( ?volition < 0 )
                 }
"""
