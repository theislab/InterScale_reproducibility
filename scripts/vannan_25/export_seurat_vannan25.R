# Export the Vannan et al. 2025 (Nature Genetics, GEO GSE250346) Seurat object to plain files
# that notebooks/vannan_25/0_setup_vannan25.ipynb assembles into an AnnData.
#
# Works with base R + Matrix only: the Seurat/SeuratObject packages are NOT needed, because the
# S4 slots are read via attributes() instead of the @ accessor (which needs the class definition).
#
# Usage:
#   Rscript scripts/vannan_25/export_seurat_vannan25.R <path/to/GSE250346_Seurat_*.rds> <out_dir>
#
# Writes to <out_dir>:
#   counts_{i,p,x}.bin, counts_dims.txt   raw RNA counts, CSC (genes x cells) == CSR (cells x genes)
#   niche_{i,p,x}.bin, niche_dims.txt     "niche" assay: neighbourhood cell-type composition counts
#   genes.txt, niche_features.txt         row names of the two assays
#   obs.csv                               cell metadata (row names = cell barcodes)
#   umap.csv                              published UMAP

suppressMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
rds_path <- args[1]
out_dir <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_csc <- function(m, prefix) {
  m <- as(m, "CsparseMatrix")
  stopifnot(all(m@x == round(m@x)))
  writeBin(as.integer(m@i), file.path(out_dir, paste0(prefix, "_i.bin")), size = 4)
  writeBin(as.integer(m@p), file.path(out_dir, paste0(prefix, "_p.bin")), size = 4)
  writeBin(as.integer(m@x), file.path(out_dir, paste0(prefix, "_x.bin")), size = 4)
  writeLines(as.character(c(nrow(m), ncol(m), length(m@x))), file.path(out_dir, paste0(prefix, "_dims.txt")))
  message(prefix, ": ", nrow(m), " x ", ncol(m), ", nnz = ", length(m@x))
}

message("Reading ", rds_path)
obj <- attributes(readRDS(rds_path))
md <- obj$meta.data

rna <- attributes(obj$assays$RNA)
stopifnot(identical(colnames(rna$counts), rownames(md)))
write_csc(rna$counts, "counts")
writeLines(rownames(rna$counts), file.path(out_dir, "genes.txt"))

niche <- attributes(obj$assays$niche)
stopifnot(identical(colnames(niche$counts), rownames(md)))
write_csc(niche$counts, "niche")
writeLines(rownames(niche$counts), file.path(out_dir, "niche_features.txt"))

write.csv(md, file.path(out_dir, "obs.csv"), quote = TRUE)

umap <- attributes(obj$reductions$umap)$cell.embeddings
# rownames here carry a stray names() attribute, so align by value rather than identical()
umap <- umap[match(rownames(md), unname(rownames(umap))), , drop = FALSE]
stopifnot(!anyNA(umap), all(unname(rownames(umap)) == rownames(md)))
write.csv(umap, file.path(out_dir, "umap.csv"), quote = TRUE)

message("Done: ", out_dir)
