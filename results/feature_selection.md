# Text-Aware Feature Selection

- Target: `binary`
- Samples: `3023`
- Original tabular features: `224`
- Text columns combined into row documents: `147`
- Numeric columns kept separately: `77`
- TF-IDF feature count: `3603`
- Combined feature count before selection: `3680`
- Selected feature count: `165`
- Selected text features: `94`
- Selected numeric features: `71`

## Selected-model performance

- Accuracy: `0.7174`
- Balanced accuracy: `0.7535`
- AUC-ROC: `0.8067`

## Top selected features

| feature | importance | feature_type |
| --- | --- | --- |
| text__tude_universitaires_suprieures demographic__5_years_of_education_calculate_the_total_number_of_years_starting_from_grade_1_o | 7.395703 | text |
| text__demographic__13_do_you_understand_french_well_enough_to_participate_in_the_study_13_comprenez yes | 6.840634 | text |
| text__tremor | 6.107169 | text |
| text__1330 | 5.892516 | text |
| text__sans_objet medication__9_non_pd_medications_9_m_dicaments_non_pd | 5.759184 | text |
| text__clinical__17_can_you_engage_in_activities_outside_of_the_home_unaccompanied_17_pouvez_vous no | 4.727837 | text |
| text__clibataire_jamais_mari demographic__3_what_is_your_current_living_situation_3_quelle_est_votre_situation_de_m_nage_a | 4.000292 | text |
| text__16 demographic__6_what_is_your_current_employment_status_6_quelle_est_votre_situation_d_emploi | 3.851558 | text |
| text__lenteur_de_mouvement_ou_sensation_de_faiblesse_bradykinsie | 3.292807 | text |
| text__clinical__4_what_were_the_first_symptoms_4_quels_taient_les_premiers_sympt_mes_de_la_malad slowness_of_movement_or_weak_feeling_bradykinesia | 2.555666 | text |
| text__not_applicable | 2.337794 | text |
| text__rigidit_musculaireslowness_of_movement_or_weak_feeling_bradykinesia lenteur_de_mouvement_ou_sensation_de_faiblesse_bradykinsie | 2.171446 | text |
| text__antidepressants | 2.16047 | text |
| text__4_severe_i_usually_use_the_support_of_another_persons_to_walk_safely_without_falling mds_updrs__2_13_freezing_over_the_past_week_on_your_usual_day_when_walking_do_you_suddenly_ | 2.097933 | text |
| text__aucun | 1.925316 | text |
| text__troubles_de_la_marcheother | 1.881399 | text |
| text__hypotension_artrielleother autre | 1.743746 | text |
| text__strong1_slight minime_ | 1.735798 | text |
| text__services_de_soutien_en_sant_mentale_travailleureuse_social_ou_thrapeuteexercise_program | 1.723582 | text |
| text__sans_objet clinical__19_was_the_patient_on_or_off_during_the_study_visit_19_le_patient_tait_il_on_ou_ | 1.657909 | text |
| text__epidemiological__8_do_you_feel_or_have_people_told_you_that_your_sense_of_smell_is_getting_worse_ uncertain | 1.655613 | text |
| text__masculin moca__was_the_moca_administered_le_moca_a_t_il_t_administr | 1.573212 | text |
| text__strong2_mild | 1.550102 | text |
| text__fminin moca__was_the_moca_administered_le_moca_a_t_il_t_administr | 1.498606 | text |
| text__oui medication__3_current_comt_inhibitor_medications_catechol_o_methyl_transferase_3_m_dication_ | 1.490223 | text |