library(readr)
data_join <- read_csv("dual_morality_RL/data/cleaned_data_join.csv")
data_model <- read_csv("dual_morality_RL/data/cleaned_data_model.csv")
data_gen <- read_csv("dual_morality_RL/data/data_generated.csv")

View(data_join)

library(lme4)
library(simr)

mixed.datajoin1 <- lmer(score_dif ~ (1|userid) + (1|gridnum), data = data_join)

mixed.datajoin2 <- lmer(score_dif ~ time_constraint  +  (1|userid) + (1|gridnum), data = data_join)

mixed.datajoin3 <- lmer(score_dif ~ time_constraint + in_distrib  +  (1|userid) + (1|gridnum), data = data_join)

mixed.datajoin4 <- lmer(score_dif ~ time_constraint + in_distrib + push +  (1|userid) + (1|gridnum), data = data_join)

mixed.datajoin5 <- lmer(score_dif ~ time_constraint + in_distrib + push + time_x_distrib +  (1|userid) + (1|gridnum), data = data_join)

mixed.datajoin6 <- lmer(score_dif ~ time_constraint + in_distrib + push + time_x_distrib + time_x_push + (1|userid) + (1|gridnum), data = data_join)
summary(mixed.datajoin6)

mixed.datajoin6b <- lmer(score_dif ~ time_constraint + in_distrib + push + time_x_push + (1|userid) + (1|gridnum), data = data_join)
summary(mixed.datajoin6b)

anova(mixed.datajoin1,mixed.datajoin2,mixed.datajoin3,mixed.datajoin4,mixed.datajoin5,mixed.datajoin6)
anova(mixed.datajoin4,mixed.datajoin5)
anova(mixed.datajoin4,mixed.datajoin6b)


sim_treat <- powerSim(mixed.datajoin, nsim=100, test = fcompare(score_dif~time_x_distrib))
sim_treat

model_ext_users <- extend(mixed.datajoin, along="userid", n=120)
p_curve <- powerCurve(model_ext_users, test=fcompare(score_dif~time_x_push), along="userid")
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
