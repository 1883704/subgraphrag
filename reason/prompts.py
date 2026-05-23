sys_prompt = (
    "Based on the triplets from a knowledge graph, please answer the given question. "
    "Please keep the answers as simple as possible and return all the possible answers "
    """as a list, each with a prefix "ans:"."""
)
cot_prompt = (
    'Format your above answers by listing each answer on a separate line, starting with the prefix "ans:".'
    # " Put your most confident answer first."
    # "2. For each answer, provide reasoning and a likelihood score based on the given triplets."
    # "3. Do not add any new answers."
    # "3. Remove irrelevant answers according to your knowledge, but do not add any additional answers."
)


sys_prompt_gpt = (
    "Based on the triplets retrieved from a knowledge graph, please select relevant triplets for answering the question."
    ' Please return formatted triplets as a list, each prefixed with "evidence:".'
)
cot_prompt_gpt = cot_prompt


sys_prompt_rm_rank = sys_prompt
cot_prompt_rm_rank = (
    "Format your above answers as follows:\n"
    "First, remove wrong answers according to your knowledge.\n"
    'Second, list each answer on a separate line, starting with the prefix "ans:".\n'
    "Finally, put your most confident answer first."
)


sys_prompt_simp = (
    "Based on the triplets from a knowledge graph, please answer the given question."
)
cot_prompt_simp = (
    'Format your answers by listing each answer on a separate line, starting with the prefix "ans:".'
)


# cot_prompt = (
#     "Format your above answers as follows: "
#     '1. List each answer on a separate line, starting with the prefix "ans:". '
#     "2. Put your most confident answer first. "
#     "3. Remove wrong answers according to your knowledge. "
#     "4. If no answer was found from the given triplets, provide answers based on your knowledge, "
#     'each with a prefix "ans:" as well.'
# )


# cot_prompt = (
#     "Format your above answers as follows: "
#     '1. List each answer on a separate line, starting with the prefix "ans:". '
#     "2. Put your most confident answer first. "
#     "3. Remove irrelevant answers according to your knowledge, but do not add any additional answers."
# )


# cot_prompt = (
#     "Format your above answers as follows: "
#     "First, remove irrelevant and wrong answers according to the triplets and your knowledge. "
#     "Second, put your most confident answer first. "
#     'Third, list each answer on a separate line, starting with the prefix "ans:". '
# )

icl_sys_prompt = (
    "Based on the triplets retrieved from a knowledge graph, please answer the question."
    ' Please return formatted answers as a list, each prefixed with "ans:".'
)

# icl_cot_prompt = (
#     # "If no answer is found, please provide an answer according to your common sense. "
#     # "Please keep the answer as simple as possible."
#     #  'Format your answers by listing each answer on a separate line, starting with the prefix "ans:".'
#     #  " Use the entity names as they appear in the knowledge graph as your answers."
#     #  f" Again, the question is: {prompts['question']}"
#     # 'If there is no sufficient information to answer the question, return "ans: not available".'
#     "Let's think step by step."
#     ' Return the most possible answers by listing each answer on a separate line, starting with the prefix "ans:".'
#         # 'Format your above answers by listing each answer on a separate line, starting with the prefix "ans:".'
#     ' Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'

# #    'If there is no sufficient information to answer the question, return "ans: not available".'
# #    'Otherwise, return the most possible answers, each prefixed with "ans:".'

# )

icl_cot_prompt = (
    # "If no answer is found, please provide an answer according to your common sense. "
    # "Please keep the answer as simple as possible."
    #  'Format your answers by listing each answer on a separate line, starting with the prefix "ans:".'
    #  " Use the entity names as they appear in the knowledge graph as your answers."
    #  f" Again, the question is: {prompts['question']}"
    # 'If there is no sufficient information to answer the question, return "ans: not available".'
    "Let's think step by step."
    ' Return the most possible answers based on the given triplets by listing each answer on a separate line, starting with the prefix "ans:".'
        # 'Format your above answers by listing each answer on a separate line, starting with the prefix "ans:".'
    ' Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'

#    'If there is no sufficient information to answer the question, return "ans: not available".'
#    'Otherwise, return the most possible answers, each prefixed with "ans:".'

)

icl_cot_prompt_post = (
    # "If no answer is found, please provide an answer according to your common sense. "
    # "Please keep the answer as simple as possible."
    #  'Format your answers by listing each answer on a separate line, starting with the prefix "ans:".'
    #  " Use the entity names as they appear in the knowledge graph as your answers."
    #  f" Again, the question is: {prompts['question']}"
    # 'If there is no sufficient information to answer the question, return "ans: not available".'
    'Return the most possible answers based on the given triplets by listing each answer on a separate line, starting with the prefix "ans:".'
        # 'Format your above answers by listing each answer on a separate line, starting with the prefix "ans:".'
    ' Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'
    " Let's think step by step."

#    'If there is no sufficient information to answer the question, return "ans: not available".'
#    'Otherwise, return the most possible answers, each prefixed with "ans:".'

)


icl_user_prompt = """Triplets:
(Lou Seal,sports.mascot.team,San Francisco Giants)
(San Francisco Giants,sports.sports_team.championships,2012 World Series)
(San Francisco Giants,sports.sports_championship_event.champion,2014 World Series)
(San Francisco Giants,time.participant.event,2014 Major League Baseball season)
(San Francisco Giants,time.participant.event,2010 World Series)
(San Francisco Giants,time.participant.event,2010 Major League Baseball season)
(San Francisco Giants,sports.sports_team.championships,2014 World Series)
(San Francisco Giants,sports.sports_team.team_mascot,Crazy Crab)
(San Francisco Giants,sports.sports_team.championships,2010 World Series)
(San Francisco Giants,sports.professional_sports_team.owner_s,Bill Neukom)
(San Francisco Giants,time.participant.event,2012 World Series)
(San Francisco,sports.sports_team_location.teams,San Francisco Giants)
(San Francisco Giants,sports.sports_team.arena_stadium,AT&T Park)
(AT&T Park,location.location.events,2012 World Series)
(m.011zsc4_,organization.leadership.organization,San Francisco Giants)
(San Francisco Giants,sports.sports_team.previously_known_as,New York Giants)
(AT&T Park,location.location.events,2010 World Series)
(Crazy Crab,sports.mascot.team,San Francisco Giants)
(New York Giants,baseball.baseball_team.league,National League)
(San Francisco Giants,sports.sports_team.colors,Black)
(San Francisco Giants,sports.sports_team.previously_known_as,New York Gothams)
(m.0k079qm,base.schemastaging.team_training_ground_relationship.team,San Francisco Giants)
(m.0k079ry,base.schemastaging.team_training_ground_relationship.team,San Francisco Giants)
(2010 World Series,time.event.locations,AT&T Park)
(San Francisco Giants,time.participant.event,2012 Major League Baseball season)
(San Francisco Giants,baseball.baseball_team.league,National League)
(m.0crtd80,sports.sports_league_participation.league,National League West)
(San Francisco Giants,sports.sports_team.location,San Francisco)
(San Francisco Giants,sports.sports_team.sport,Baseball)
(m.05n6dtn,baseball.baseball_team_stats.team,San Francisco Giants)


Question:
What year did the team with mascot named Lou Seal win the World Series?"""

icl_ass_prompt = """To find the year the team with mascot named Lou Seal won the World Series, we need to find the team with mascot named Lou Seal and then find the year they won the World Series.

From the triplets, we can see that Lou Seal is the mascot of the San Francisco Giants.

Now, we need to find the year the San Francisco Giants won the World Series.

From the triplets, we can see that San Francisco Giants won the 2010 World Series and 2012 World Series and 2014 World Series.

So, the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012, and 2014.

Therefore, the formatted answers are:

ans: 2014 (2014 World Series)
ans: 2012 (2012 World Series)
ans: 2010 (2010 World Series)"""


noevi_sys_prompt = (
    "Please answer the question."
    ' Please return formatted answers as a list, each prefixed with "ans:".'
)


noevi_cot_prompt = (
    # "If no answer is found, please provide an answer according to your common sense. "
    # "Please keep the answer as simple as possible."
    #  'Format your answers by listing each answer on a separate line, starting with the prefix "ans:".'
    #  " Use the entity names as they appear in the knowledge graph as your answers."
    #  f" Again, the question is: {prompts['question']}"
    # 'If there is no sufficient information to answer the question, return "ans: not available".'
    "Let's think step by step."
    ' Return the most possible answers by listing each answer on a separate line, starting with the prefix "ans:".'
        # 'Format your above answers by listing each answer on a separate line, starting with the prefix "ans:".'
    ' Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'

#    'If there is no sufficient information to answer the question, return "ans: not available".'
#    'Otherwise, return the most possible answers, each prefixed with "ans:".'

)


mcq_sys_prompt = (
    "You are answering a four-option single-answer medical question. "
    "The question includes four options labeled A, B, C, and D. "
    "You may use the provided knowledge-graph evidence if it is relevant. "
    "If the evidence is incomplete or irrelevant, rely on your medical knowledge and the option text. "
    "Choose exactly one option from A, B, C, or D. "
    "Return only one line in this exact format: ans: <letter>. "
    "Do not explain. Do not output option text. Do not output any other characters."
)


mcq_cot_prompt = (
    "Choose exactly one option from A, B, C, or D. "
    "Return only one line in this exact format: ans: A, ans: B, ans: C, or ans: D."
)


mcq_gate_sys_prompt = (
    "You are answering a four-option single-answer medical question. "
    "The question includes options A, B, C, and D. "
    "Reasoning trees from a medical knowledge graph may be provided, but they are noisy. "
    "Use a reasoning tree only when it directly supports or rules out an option. "
    "Ignore trees built from generic entities such as all, disease, symptom, disorder, patient, or person. "
    "Ignore evidence that is only loosely related to the question topic or only repeats an option without a useful relation. "
    "If the trees are incomplete, generic, contradictory, or clinically irrelevant, rely on the question and your medical knowledge. "
    "Choose exactly one option from A, B, C, or D. "
    "Return only one line in this exact format: ans: <letter>. "
    "Do not explain. Do not output option text. Do not output any other characters."
)


mcq_gate_cot_prompt = (
    "Decide whether the provided reasoning trees are clinically relevant. "
    "Then choose exactly one option from A, B, C, or D. "
    "Return only one line in this exact format: ans: A, ans: B, ans: C, or ans: D."
)


mcq_compare_sys_prompt = (
    "You are answering a four-option single-answer medical question. "
    "The question includes options A, B, C, and D. "
    "Compare the four option texts against the question first. "
    "Then use the provided reasoning trees only as supporting evidence when their entities and relations are specific to the option being evaluated. "
    "Treat generic or hub-like evidence as low confidence. "
    "Do not let irrelevant knowledge-graph paths override clear medical knowledge. "
    "Choose exactly one option from A, B, C, or D. "
    "Return only one line in this exact format: ans: <letter>. "
    "Do not explain. Do not output option text. Do not output any other characters."
)


mcq_compare_cot_prompt = (
    "Compare options A, B, C, and D, using reasoning trees only if they are specific and relevant. "
    "Return only one line in this exact format: ans: A, ans: B, ans: C, or ans: D."
)
