library(jsonlite)
library(HiCcompare)

# Define root directories and chromosomes
root_dirs <- c("Src")
chromosomes <- c("chr1", "chr3", "chr5", "chr7", "chr8", "chr9", "chr11", "chr13", "chr15", "chr17", "chr18", "chr19", "chr21", "chr22")


# Define the base path where the root directories are located
base_path <- "~/DNase_Data/HMEC"

# Loop through each root directory and chromosome
for (root in root_dirs) {
  for (chr in chromosomes) {
    # Set the current working directory
    root_chr_path <- file.path(base_path, root, chr)
    setwd(root_chr_path)
    
    args <- c(paste0(chr, "_Outputs"), chr, "Genomic_fragment")
    
    # Preprocessing
    curr_dir <- getwd()
    dir.create(args[1], showWarnings = FALSE)
    
    raw_count_file <- paste0(args[2], "_10kb.RAWobserved")
    kr_file_path <- paste0(args[2], "_10kb.KRnorm")
    
    
    if (file.exists(raw_count_file) & file.exists(kr_file_path)) {
      raw_count <- read.table(raw_count_file)
      kr_file <- read.table(kr_file_path)
    } else {
      print(paste("Files missing in directory:", curr_dir))
      next
    }
    
    n <- floor(max(raw_count[, c(1, 2)])) / 5e3 + 1
    normalized_mat <- matrix(0, n, n)
    
    i_ind <- floor(raw_count[, 1] / 5e3) + 1
    j_ind <- floor(raw_count[, 2] / 5e3) + 1
    
    for (i in 1:dim(raw_count)[1]) {
      tmp_i <- i_ind[i]
      tmp_j <- j_ind[i]
      normalized_mat[tmp_i, tmp_j] <- raw_count[i, 3] / (kr_file[tmp_i, ] * kr_file[tmp_j, ])
    }
    
    input_if <- normalized_mat
    input_if <- input_if + t(input_if)
    diag(input_if) <- diag(input_if) / 2
    
    pd <- input_if^(-1/4)
    setwd(curr_dir)
    save(input_if, file = paste0(args[1], "/", args[2], "_10kb_IF.Rdata"))
    save(pd, file = paste0(args[1], "/", args[2], "_10kb_PD.Rdata"))
    
    #  Generate Fragment Data
    nextdir <- paste0(getwd(), "/", args[1])
    setwd(nextdir)
    
    if (file.exists(paste0(chr, "_10kb_IF.Rdata")) & file.exists(paste0(chr, "_10kb_PD.Rdata"))) {
      load(paste0(chr, "_10kb_IF.Rdata"))
      load(paste0(chr, "_10kb_PD.Rdata"))
    } else {
      print(paste("Processed files missing for chromosome:", chr))
      stop()
    }
    
    n <- ceiling(dim(input_if) / 200)[1]
    dir.create(paste0(chr, "_10kb_frag"), showWarnings = FALSE)
    setwd(paste0(chr, "_10kb_frag"))
    
    for (i in 1:n) {
      start_id <- (i - 1) * 200 + 1
      end_id <- min(dim(pd)[1], i * 200)
      
      tmp_pd <- pd[start_id:end_id, start_id:end_id]
      tmp_input_if <- input_if[start_id:end_id, start_id:end_id]
      write.table(as.matrix(tmp_pd), paste0("Dist_frag", i, '.txt'), col.names = FALSE, row.names = FALSE, sep = "\t", quote = FALSE)
      write.table(as.matrix(tmp_input_if), paste0("IF_frag", i, '.txt'), col.names = FALSE, row.names = FALSE, sep = "\t", quote = FALSE)
    }
    
    #  Generate Fragment DNase
    newwdir <- paste0(curr_dir,"/", args[1])
    setwd(newwdir)
    
    if (file.exists(paste0(chr, "_10kb_PD.Rdata"))) {
      load(paste0(chr, "_10kb_PD.Rdata"))
    } else {
      print(paste("Processed PD file missing for chromosome:", chr))
      next
    }
    
    options(scipen = 999)
    N <- floor(dim(pd)[1] / 200)
    dir.create('./Genomic_fragment/', showWarnings = FALSE)
    
    for (i in 1:N) {
      start_id <- (i - 1) * 200 + 1
      end_id <- min(dim(pd)[1], i * 200)
      start_loc <- (seq(start_id, end_id) - 1) * 5000 + 1
      end_loc <- start_loc + 4999
      frag_list <- data.frame(chr = chr, start = start_loc, end = end_loc)
      write.table(frag_list, paste0("./Genomic_fragment/Genomic_frag", i, '.txt'), col.names = FALSE, row.names = FALSE, sep = "\t", quote = FALSE)
    }
    
    # Generate DNase Profile
    setwd(paste0(curr_dir, "/", args[1], "/Genomic_fragment"))
    dnase_file <- paste0(base_path, "/", root, "/", chr, "/", chr, ".bed")
    
    if (!file.exists(dnase_file)) {
      print(paste("DNase file missing:", dnase_file))
      next
    }
    
    file <- dir()
    file <- file[grep("Genomic_frag", file)]
    n <- length(file)
    
    total_file <- data.frame()
    
    for (i in 1:n) {
      tmp_file <- read.table(paste0('Genomic_frag', i, ".txt"))
      total_file <- rbind(total_file, tmp_file)
    }
    
    write.table(total_file, "total_file.txt", col.names = FALSE, sep = '\t', row.names = FALSE, quote = FALSE)
    system("sort -k1,1 -k2,2n total_file.txt > sorted_total_file.txt")
    command <- paste("bedtools map -a sorted_total_file.txt -b", dnase_file, "-c 5 -o mean > total_Dnase.txt")
    system(command)
    
    total_DNase <- read.table('total_Dnase.txt')
    dnase_list <- split(total_DNase, floor((0:(dim(total_DNase)[1] - 1)) / 200))
    
    dir.create("DNase_Genomic_frags", showWarnings = FALSE)
    
    for (i in 1:length(dnase_list)) {
      write.table(dnase_list[[i]], paste0('./DNase_Genomic_frags/DNase_Genomic_frag', i, '.txt'), quote = FALSE, col.names = FALSE, row.names = FALSE, sep = '\t')
    }
    
    # Impute DNase Distance
    load(paste0(base_path, "/exponential_linear_model_-0.4.Rdata"))
    setwd(paste0(curr_dir, "/", args[1], "/DNase_Genomic_frags"))
    
    file <- dir()
    n <- length(grep('DNase_Genomic_frag', file))
    
    dir.create("imputed_3D_Dnase", showWarnings = FALSE)
    dirr <- "imputed_3D_Dnase"
    setwd(dirr)
    
    for (id in 1:n) {
      Dnase_signal <- read.table(paste0("./../DNase_Genomic_frag", id, ".txt"))
      N <- dim(Dnase_signal)[1]
      Dnase_mat <- matrix(0, N, N)
      
      od_vec <- abs(row(Dnase_mat) - col(Dnase_mat)) * 5000
      dnase_1 <- as.numeric(Dnase_signal[row(Dnase_mat), 4])
      dnase_2 <- as.numeric(Dnase_signal[col(Dnase_mat), 4])
      data <- data.frame(genomic_distance = as.vector(od_vec), dnase_1, dnase_2)
      pred_val <- predict(l_model, data)
      predict_dist <- matrix(pred_val, byrow = FALSE, N)
      predict_dist <- as.matrix(predict_dist)
      predict_dist <- predict_dist^2
      predict_dist[which(predict_dist == Inf)] <- 3
      predict_dist[which(is.na(predict_dist))] <- 3
      
      converted_IF <- predict_dist^(-1/4)
      write.table(converted_IF, paste0("DNase_3D_IF_impute", id, ".txt"), col.names = FALSE, row.names = FALSE, sep = '\t', quote = FALSE)
      write.table(predict_dist, paste0("DNase_3D_PD_impute", id, ".txt"), col.names = FALSE, row.names = FALSE, sep = '\t', quote = FALSE)
    }
    
    #  Normalize and Save
    setwd(paste0(curr_dir, "/", args[1], "/imputed_3D_Dnase"))
    combined_IF_matrix <- matrix(0, nrow = dim(input_if)[1], ncol = dim(input_if)[2])
    
    for (i in 1:n) {
      tmp_input_if <- as.matrix(read.table(paste0("DNase_3D_IF_impute", i, ".txt"), header = FALSE))
      start_id <- (i - 1) * 200 + 1
      end_id <- min(dim(pd)[1], i * 200)
      combined_IF_matrix[start_id:end_id, start_id:end_id] <- tmp_input_if
    }
    
    normalized_if_matrix <- KRnorm(combined_IF_matrix)
    filename <- paste0(args[2], "_10kb.tsv")
    write.table(normalized_if_matrix, file = filename, sep = "\t", col.names = FALSE, row.names = FALSE, quote = FALSE)
    
    rm(list = ls())
    gc()
  }
}

rm(list = ls())
gc()
