suppressMessages({library(metafor); library(clubSandwich)})
out <- list()
add <- function(tag, b, se, lb, ub, p, tau2, i2, pilb, piub) out[[length(out)+1]] <<- data.frame(tag, b, se, lb, ub, p, tau2, i2, pilb, piub)
for (meas in c("SMD","MD","ROM")) {
  d <- read.csv("/tmp/kt/rv/single.csv")
  e <- escalc(meas, m1i=Mean_treat, sd1i=SD_treat, n1i=N_treat, m2i=Mean_control, sd2i=SD_control, n2i=N_control, data=d)
  for (cfg in list(c("REML","knha"), c("REML","z"), c("DL","z"), c("FE","z"))) {
    r <- rma(yi, vi, data=e, method=cfg[1], test=cfg[2]); pr <- predict(r)
    pilb <- if (is.null(pr$pi.lb)) pr$ci.lb else pr$pi.lb; piub <- if (is.null(pr$pi.ub)) pr$ci.ub else pr$pi.ub
    add(paste("single", meas, cfg[1], cfg[2]), r$b[1], r$se, r$ci.lb, r$ci.ub, r$pval, r$tau2, r$I2, pilb, piub)
  }
  m <- read.csv("/tmp/kt/rv/multi.csv")
  e <- escalc(meas, m1i=Mean_treat, sd1i=SD_treat, n1i=N_treat, m2i=Mean_control, sd2i=SD_control, n2i=N_control, data=m)
  e$es_id <- seq_len(nrow(e))
  r <- rma.mv(yi, vi, random=~1|Study/es_id, data=e, method="REML", test="t"); pr <- predict(r)
  add(paste("multi", meas, "ML3_T"), r$b[1], r$se, r$ci.lb, r$ci.ub, r$pval, sum(r$sigma2), NA, pr$pi.lb, pr$pi.ub)
  ct <- coef_test(r, vcov="CR2", cluster=e$Study, test="Satterthwaite"); ci <- conf_int(r, vcov="CR2", cluster=e$Study, test="Satterthwaite")
  add(paste("multi", meas, "ML3_CR2"), r$b[1], ct$SE, ci$CI_L, ci$CI_U, ct$p_Satt, sum(r$sigma2), NA, pr$pi.lb, pr$pi.ub)
  a <- aggregate(e, cluster=Study, rho=0.6)
  r <- rma(yi, vi, data=a, method="REML", test="knha"); pr <- predict(r)
  add(paste("multi", meas, "AGG_HKSJ"), r$b[1], r$se, r$ci.lb, r$ci.ub, r$pval, r$tau2, r$I2, pr$pi.lb, pr$pi.ub)
}
write.csv(do.call(rbind, out), "/tmp/kt/rv/ref.csv", row.names=FALSE)
