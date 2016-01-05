define ultimate goal

while state != ultimate goal:

	if state != i_goal:
		find action with effects that best match i_goal
		replace i_goal with i_goal - action_effects + action_preconds (only ones that do not match state)

	elif stack not empty:
		a = peek at top of stack
		check world state (given target focus from action)
		if a preconds met:
			pop a, execute

	else:
		i_goal = ultimate_goal


-- stop planning indefinitely if too many actions are pushed to the stack. 

** plan in the absence of any info about world state, only use expectations about which preconds are likely to be met
** then, when action is popped from stack, check world state