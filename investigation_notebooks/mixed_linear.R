library(readr)
data <- read_csv("dual_morality_RL/data/cleaned_data.csv")
data_model <- read_csv("dual_morality_RL/data/cleaned_data_model.csv")
View(data_model)

library(lme4)


# for -0.5,0.5 in-distrib values, subtract mean from every individual datapt
mixed.data <- lmer(score_dif ~ time_constraint + in_distrib + time_x_distrib + (in_distrib|userid) + (time_constraint|gridnum), data = data)
summary(mixed.data)

mixed.data2 <- lmer(score_dif ~ time_constraint + in_distrib + (in_distrib|userid) + (time_constraint|gridnum), data = data)
summary(mixed.data2)

anova(mixed.data,mixed.data2)

mixed.model <- lmer(score_dif ~ time_constraint + in_distrib + time_x_distrib + (in_distrib|userid) + (time_constraint|gridnum), data = data_model)
summary(mixed.model)

mixed.model2 <- lmer(score_dif ~ time_constraint + in_distrib + (in_distrib|userid) + (time_constraint|gridnum), data = data_model)
summary(mixed.model2)

anova(mixed.model,mixed.model2)
