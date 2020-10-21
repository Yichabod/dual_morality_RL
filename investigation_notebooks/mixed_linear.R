library(readr)
data_join <- read_csv("dual_morality_RL/data/cleaned_data_join.csv")
View(data_join)

library(lme4)
library(brms)
library(simr)

#time constraint,indistrib: -0.5,0.5
#push vs switch (only out of distrib) 0 if indistrib, -0.5,0.5 for push and switch
#nested set of predictors
#list of hypotheses about what differs between switch and push
mixed.datajoin <- lmer(score_dif ~ time_constraint + in_distrib + push + time_x_distrib + time_x_push + (in_distrib|userid) + (time_constraint|gridnum), data = data_join)
summary(mixed.datajoin)

mixed.datajoin2 <- lmer(score_dif ~ time_constraint + in_distrib + push + time_x_distrib + (time_constraint|gridnum), data = data_join)
summary(mixed.datajoin2)

anova(mixed.datajoin,mixed.datajoin2)

mixed.datab <- brm(score_dif ~ time_constraint + in_distrib + push + time_x_distrib + time_x_push + (in_distrib|userid) + (time_constraint|gridnum), data = data_join)
summary(mixed.datab)

mixed.datab2 <- brm(score_dif ~ time_constraint + in_distrib + push + (in_distrib|userid) + (time_constraint|gridnum), data = data_join)
summary(mixed.datab2)

loo(mixed.datab,mixed.datab2)


sim_treat <- powerSim(mixed.datapush, nsim=100, test = fixed("time_x_distrib"))
sim_treat
p_curve <- powerCurve(mixed.datapush, test=fixed("time_x_distrib"), along="userid")
plot(p_curve)



# # for -0.5,0.5 in-distrib values, subtract mean from every individual datapt
# mixed.datapush <- lmer(score_dif ~ time_constraint + in_distrib + time_x_distrib + (in_distrib|userid) + (time_constraint|gridnum), data = data_push)
# summary(mixed.datapush)
# 
# mixed.data2 <- brm(score_dif ~ time_constraint + in_distrib + (in_distrib|userid) + (time_constraint|gridnum), data = data)
# summary(mixed.data2)
# 
# loo(mixed.data,mixed.data2)
# anova(mixed.data,mixed.data2)
# 
# mixed.dataswitch <- brms(score_dif ~ time_constraint + in_distrib + time_x_distrib + (in_distrib|userid) + (time_constraint|gridnum), data = data_switch)
# summary(mixed.dataswitch)
# 
# mixed.dataswitch2 <- lmer(score_dif ~ time_constraint + in_distrib + (in_distrib|userid) + (time_constraint|gridnum), data = data_switch)
# summary(mixed.dataswitch2)
# 
# anova(mixed.dataswitch,mixed.dataswitch2)
# 
# mixed.model <- lmer(score_dif ~ time_constraint + in_distrib + time_x_distrib + (in_distrib|userid) + (time_constraint|gridnum), data = data_model)
# summary(mixed.model)
# 
# mixed.model2 <- lmer(score_dif ~ time_constraint + in_distrib + (in_distrib|userid) + (time_constraint|gridnum), data = data_model)
# summary(mixed.model2)
# 
# anova(mixed.model,mixed.model2)
