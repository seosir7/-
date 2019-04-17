
indian = read.csv('pima-indians-diabetes.csv', header=F)


colnames(indian)=c('pregnant','plasma','pressure','thickness','insulin','BMI','pedigree','age','class')

head(indian,7)

dim(indian)
sd(indian$pregnant) # 제품_친밀도 

sd(indian$plasma) #  제품_적절성

sd(indian$pressure) #  제품_만족도 

sd(indian$thickness)

sd(indian$insulin)

sd(indian$BMI)

sd(indian$pedigree)

sd(indian$age)

indian$class

cor(indian)
cor(indian$pregnant, indian$class, method='pearson' )
cor(indian$plasma, indian$class, method='pearson' )
cor(indian$pressure, indian$class, method='pearson' )
cor(indian$thickness, indian$class, method='pearson' )
cor(indian$insulin, indian$class, method='pearson' )
cor(indian$BMI, indian$class, method='pearson' )
cor(indian$pedigree, indian$class, method='pearson' )
cor(indian$age, indian$class, method='pearson' )


corrgram( indian)
corrgram( indian, upper.panel = panel.conf)



corrgram( indian, lower.panel = panel.conf)

chart.Correlation(indian, pch='+')


corrgram(a)
?corrgram
?barplot

plot(indian)
