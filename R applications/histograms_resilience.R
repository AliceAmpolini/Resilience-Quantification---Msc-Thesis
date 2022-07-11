#create a histogram and a boxplot for pluvial flooding in Semarang
v <- (as.numeric(Di_pluv_sem[,]))
hist(v, main = "", xlab = "", breaks = 7)
par = (new=T)
boxplot(v,horizontal=TRUE,outline=FALSE, ylim=c(0,0.2))

abline(v = median(v),                     # Add line for median
       col = "red",					#Red and with width=5
       lwd = 5)
text(x = 0.09,                 # Add text for median
     y = 5.5,
     paste("Median =", round(median(v), digits = 2)),
     col = "red",
     cex = 2)

#create a histogram and a boxplot with boxplot for coastal flooding in Semarang

v1 <- (as.numeric(unlist(DI_coastal[,])))

hist(v1, main = "", xlab = "", breaks = 7)

boxplot(v1,horizontal=TRUE,outline=FALSE, ylim=c(0,0.8))

abline(v = median(v1),                     # Add line for median
       col = "red",					#Red and with width=5
       lwd = 5)
text(x = 0.2,                 # Add text for median
     y = 3.5,
     paste("Median =", round(median(v1), digits = 2)),
     col = "red",
     cex = 2)

#create a histogram and a boxplot for pluvial flooding in Vientiane

vp <- (as.numeric(unlist(Vientiane_pluvial_di)))
hist(vp, main = "", xlab = "", xlim = c(0.05,0.3), breaks = 7)
boxplot(vp, horizontal = TRUE, outline = FALSE, ylim = c(0.05,0.3))

abline(v = median(vp),                     # Add line for median
       col = "red",					#Red and with width=5
       lwd = 5)
text(x = 0.21,                 # Add text for median
     y = 1.5,
     paste("Median =", round(median(vp), digits = 2)),
     col = "red",
     cex = 2)

#create a histogram and a boxplot for fluvial flooding in Vientiane

vf <- (as.numeric(unlist(vientiane_di_fluv[,])))

hist(vf, main = "", xlab = "", breaks = 7, xlim = c(0,0.5))

boxplot(vf, horizontal = TRUE, outline = FALSE, ylim = c(0,0.5))

abline(v = median(vf),                     # Add line for median
       col = "red",					#Red and with width=5
       lwd = 5)
text(x = 0.21,                 # Add text for median
     y = 1.5,
     paste("Median =", round(median(vf), digits = 2)),
     col = "red",
     cex = 2)

#create histogram and boxplot for semarang pluvial resilience 
rsp <- (as.numeric(unlist(res_sem_pluv[,])))
par(mfrow=c(1,1))
hist(rsp, main ="Semarang Pluvial Flooding", xlab = "", breaks = 10, xlim = c(0,1))
                                   # Draw histogram
abline(v = mean(rsp),                       # Add line for mean
       col = "red",
       lwd = 3)
text(x = 0.85,                   # Add text for mean
     y = 15,
     paste("Mean =", round(mean(rsp), digits = 2)),
     col = "red",
     cex = 2)
t.test(rsp)

#create histogram and boxplot for semarang coastal resilience
rsc <- (as.numeric(unlist(res_sem_coast[,])))

hist(rsc, main = "Semarang Coastal Flooding", xlab = "", breaks = 10, xlim = c(0,1))


abline(v = mean(rsc),                       # Add line for mean
       col = "red",
       lwd = 3)
text(x = 0.88,                   # Add text for mean
     y = 15,
     paste("Mean =", round(mean(rsc), digits = 2)),
     col = "red",
     cex = 2)

t.test(rsc)

#create histogram and boxplot for vientiane pluvial resilience
rvp <- (as.numeric(unlist(res_vient_pluv_vero[,])))

hist(rvp, main = "Vientiane Pluvial Flooding ", xlab = "", breaks = 10, xlim = c(0,1))
abline(v = mean(rvp),                       # Add line for mean
       col = "red",
       lwd = 3)
text(x = 0.85,                   # Add text for mean
     y = 15,
     paste("Mean =", round(mean(rvp), digits = 2)),
     col = "red",
     cex = 2)

t.test(rvp)

#create histogram and boxplot for vientiane fluvial resilience
rvf <- (as.numeric(unlist(res_vient_pluv[,])))
rvf2 <- seq(min(rvf), max(rvf), length = 101)
hist(rvf, main = "Vientiane Fluvial Flooding ", xlab = "", breaks = 10, xlim = c(0,1))
abline(v = mean(rvf),                       # Add line for mean
       col = "red",
       lwd = 3)
text(x = 0.55,                   # Add text for mean
     y = 15,
     paste("Mean =", round(mean(rvf), digits = 2)),
     col = "red",
     cex = 2)

t.test(rvf)

fun <- dnorm(rvf2, mean = mean (rvf), sd = sd(rvf))
hist(rvf, col = "white", xlim = c(0,1), ylim = c(0,20), main = "", breaks = 10)
lines (rvf2, fun, col = 2, lwd = 2)

#creating theoretical normally distributed curves of resilience for the case studies

rvf2 <- seq(min(rvf), max(rvf), length = 101)
fun <- dnorm(rvf2, mean = mean (rvf), sd = sd(rvf))
plot(rvf2,fun, type = "l", col = "darkred", lwd = 3, xlim = c(0,1), ylim = c(0,4.2))
text(x = 0.39,                  
     y = 3,
     paste("(d)"),
     col = "darkred",
     cex = 2)
     
rvp2 <- seq(min(rvp), max(rvp), length = 101)
fun <- dnorm(rvp2, mean = mean (rvp), sd = sd(rvp))
lines(rvp2,fun, type = "l", col = "red", lwd = 3, xlab = "", ylab = "")
text(x = 0.55,                  
     y = 4,
     paste("(c)"),
     col = "red",
     cex = 2)

rsp2 <- seq(min(rsp), max(rsp), length = 101)
fun <- dnorm(rsp2, mean = mean (rsp), sd = sd(rsp))
lines(rsp2,fun, type = "l", col ="gold", lwd = 3, xlab = "", ylab = "")
text(x = 0.65,                  
     y = 3.2,
     paste("(a)"),
     col = "gold",
     cex = 2)

rsc2 <- seq(min(rsc), max(rsc), length = 101)
fun <- dnorm(rsc2, mean = mean (rsc), sd = sd(rsc))
lines(rsc2,fun, type = "l", col = "green", lwd = 3, xlab = "", ylab = "")
text(x = 0.68,                  
     y = 2.5,
     paste("(b)"),
     col = "green",
     cex = 2)



