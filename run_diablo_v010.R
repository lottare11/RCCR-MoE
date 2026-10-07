suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(mixOmics))
options(rgl.useNULL=TRUE)
root <- "data/diablo_folds"
outdir <- "results/baseline_rebuild_v010_20260922/multiomics/diablo"
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)

auc_rank <- function(y,s){
  y <- as.integer(y)
  n1 <- sum(y==1); n0 <- sum(y==0)
  r <- rank(s,ties.method="average")
  (sum(r[y==1]) - n1*(n1+1)/2)/(n1*n0)
}
balacc <- function(y,p,th){
  z <- as.integer(p>=th)
  sens <- mean(z[y==1]==1)
  spec <- mean(z[y==0]==0)
  (sens+spec)/2
}
best_threshold <- function(y,p){
  ths <- seq(0.05,0.95,by=0.005)
  sc <- sapply(ths,function(t) balacc(y,p,t))
  ths[which.max(sc)]
}
readm <- function(fd,part,mod){
  as.matrix(read.csv(file.path(fd,paste0(part,"_",mod,".csv")),check.names=FALSE))
}
configs <- expand.grid(ncomp=c(1,2),kid=1:4)
keep_sets <- list(c(50,50,32),c(100,100,64),c(200,100,64),c(200,200,128))
all_oof <- data.frame()
all_meta <- data.frame()
for(fold in 0:4){
  fd <- file.path(root,paste0("fold_",fold))
  trX <- list(SERS=readm(fd,"train","sers"),Transcriptome=readm(fd,"train","tran"),Metabolome=readm(fd,"train","meta"))
  vaX <- list(SERS=readm(fd,"val","sers"),Transcriptome=readm(fd,"val","tran"),Metabolome=readm(fd,"val","meta"))
  teX <- list(SERS=readm(fd,"test","sers"),Transcriptome=readm(fd,"test","tran"),Metabolome=readm(fd,"test","meta"))
  trlab <- read.csv(file.path(fd,"train_labels.csv"))
  valab <- read.csv(file.path(fd,"val_labels.csv"))
  telab <- read.csv(file.path(fd,"test_labels.csv"))
  Y <- factor(trlab$y,levels=c(0,1))
  design <- matrix(0.1,3,3); diag(design) <- 0

  best_auc <- -Inf
  best_cfg <- NULL
  best_model <- NULL
  best_vp <- NULL
  for(ii in 1:nrow(configs)){
    nc <- configs$ncomp[ii]
    ks <- keep_sets[[configs$kid[ii]]]
    kx <- list(SERS=rep(ks[1],nc),Transcriptome=rep(ks[2],nc),Metabolome=rep(ks[3],nc))
    m <- block.splsda(trX,Y,ncomp=nc,keepX=kx,design=design)
    pr <- predict(m,newdata=vaX)
    vp <- as.numeric(pr$WeightedPredict[,"1",nc])
    a <- auc_rank(valab$y,vp)
    if(a > best_auc + 1e-12){
      best_auc <- a
      best_cfg <- c(ncomp=nc,sers=ks[1],tran=ks[2],meta=ks[3])
      best_model <- m
      best_vp <- vp
    }
  }
  th <- best_threshold(valab$y,best_vp)
  nc <- as.integer(best_cfg["ncomp"])
  prt <- predict(best_model,newdata=teX)
  tp <- as.numeric(prt$WeightedPredict[,"1",nc])
  pred <- as.integer(tp>=th)

  oof <- data.frame(method="DIABLO",subject_id=telab$subject_id,fold=fold,true_label=telab$y,
                    predicted_probability=tp,predicted_label=pred,threshold=th)
  all_oof <- rbind(all_oof,oof)
  all_meta <- rbind(all_meta,data.frame(
    fold=fold,train_size=nrow(trX$SERS),val_size=nrow(vaX$SERS),test_size=nrow(teX$SERS),
    sers_dim=ncol(trX$SERS),tran_dim=ncol(trX$Transcriptome),meta_dim=ncol(trX$Metabolome),
    best_val_auc=best_auc,ncomp=best_cfg["ncomp"],keepX_sers=best_cfg["sers"],
    keepX_tran=best_cfg["tran"],keepX_meta=best_cfg["meta"],threshold=th))
  cat("DIABLO fold",fold,"valAUC",best_auc,"cfg",paste(best_cfg,collapse="/"),"threshold",th,"\n")
}
write.csv(all_oof,file.path(outdir,"oof_predictions.csv"),row.names=FALSE)
write.csv(all_meta,file.path(outdir,"fold_metadata.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(outdir,"r_session_info.txt"))
