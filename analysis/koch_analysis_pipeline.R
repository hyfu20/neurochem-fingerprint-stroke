#!/usr/bin/env Rscript
# ==============================================================================
# Koch (2025, Brain) 金标准统计分析流水线
# 递质特异性损伤残差 × 临床指标（IL-6, HRV）关联分析
#
# 作者: 刘正鑫实验室 | 3582 人大数据分析
# 日期: 2026-03-19
#
# 用法:
#   在 RStudio 中逐段运行, 或命令行:
#   Rscript koch_analysis_pipeline.R
#
# 输入:
#   1. NT_Imaging_Load_Master.csv   (影像提取的 17 通路负荷)
#   2. variablelist.CSV              (临床变量表)
#
# 输出:
#   results_koch_R/
#     master_merged.csv             — 合并后的完整数据
#     residuals_table.csv           — 残差数据
#     regression_summary.csv        — 回归模型摘要
#     fingerprint_IL6.csv           — IL-6 指纹图谱数据
#     fingerprint_HRV.csv           — HRV 指纹图谱数据
#     fingerprint_all_clinical.csv  — 全临床变量指纹
#     fig_fingerprint_IL6.pdf/png   — IL-6 指纹柱状图
#     fig_fingerprint_HRV.pdf/png   — HRV 指纹柱状图
#     fig_load_vs_tlv.pdf/png       — Load ~ TLV 散点图
#     fig_heatmap.pdf/png           — 相关矩阵热图
# ==============================================================================

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  0. 环境准备                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

cat("=" %s+% strrep("=", 68) %s+% "\n")
cat("  Koch (2025, Brain) 递质特异性损伤分析 — R Pipeline\n")
cat("  样本: ~3582 名卒中患者\n")
cat("=" %s+% strrep("=", 68) %s+% "\n\n")

# 安装/加载所需的包
required_pkgs <- c("tidyverse", "broom", "corrplot", "RColorBrewer",
                   "ggrepel", "scales", "patchwork")

for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  📦 安装 %s ...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
}

library(tidyverse)
library(broom)
library(corrplot)
library(scales)

# 尝试加载可选包
has_patchwork <- requireNamespace("patchwork", quietly = TRUE)
if (has_patchwork) library(patchwork)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  路径配置 — 请根据你的环境修改                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 影像负荷数据
IMG_CSV  <- "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/NT_Imaging_Load_Master.csv"
# 临床变量数据
CLIN_CSV <- "/data/usersdir/liuzhengxin/Variable_list/variablelist.CSV"
# 输出目录
OUT_DIR  <- "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/results_koch_R"

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
fig_dir <- file.path(OUT_DIR, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. 读取与缝合数据                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

cat("\n[1] 读取数据...\n")

# 读取影像数据
# check.names = FALSE 防止 R 自动把 "5HT1a" 改成 "X5HT1a"
img_df <- read.csv(IMG_CSV, check.names = FALSE, stringsAsFactors = FALSE)
cat(sprintf("  影像数据: %d 行 × %d 列\n", nrow(img_df), ncol(img_df)))
cat(sprintf("  列名: %s\n", paste(colnames(img_df), collapse = ", ")))

# 读取临床数据
# 尝试不同编码 (Windows 中文 CSV 常用 GBK)
tryCatch({
  clin_df <- read.csv(CLIN_CSV, check.names = FALSE, stringsAsFactors = FALSE,
                      fileEncoding = "UTF-8")
}, error = function(e) {
  clin_df <<- read.csv(CLIN_CSV, check.names = FALSE, stringsAsFactors = FALSE,
                       fileEncoding = "GBK")
})
cat(sprintf("  临床数据: %d 行 × %d 列\n", nrow(clin_df), ncol(clin_df)))
cat(sprintf("  前20列: %s\n", paste(head(colnames(clin_df), 20), collapse = ", ")))

# --- 智能 ID 匹配 ---
# 找到两表中的 ID 列 (可能叫 ID, id, Subject, PatientID 等)
find_id_col <- function(df) {
  candidates <- c("ID", "id", "Id", "Subject", "PatientID", "SubjectID", "SID")
  for (cand in candidates) {
    if (cand %in% colnames(df)) return(cand)
  }
  # 回退: 第一列
  return(colnames(df)[1])
}

id_col_img  <- find_id_col(img_df)
id_col_clin <- find_id_col(clin_df)
cat(sprintf("  影像 ID 列: '%s' | 临床 ID 列: '%s'\n", id_col_img, id_col_clin))

# 统一 ID 列名为 "ID"
if (id_col_img != "ID") {
  img_df <- img_df %>% rename(ID = !!sym(id_col_img))
}
if (id_col_clin != "ID") {
  clin_df <- clin_df %>% rename(ID = !!sym(id_col_clin))
}

# 确保 ID 为字符型, 去除前后空格
img_df$ID  <- trimws(as.character(img_df$ID))
clin_df$ID <- trimws(as.character(clin_df$ID))

# 合并
master_df <- inner_join(clin_df, img_df, by = "ID", suffix = c("_clin", "_img"))

cat(sprintf("\n  ✅ 合并成功! 最终样本量: %d\n", nrow(master_df)))
cat(sprintf("     影像中的 ID: %d | 临床中的 ID: %d | 交集: %d\n",
            n_distinct(img_df$ID), n_distinct(clin_df$ID), nrow(master_df)))

# 如果匹配较少, 打印诊断
if (nrow(master_df) < 100) {
  cat("\n  ⚠️  匹配数量偏少! 请检查 ID 格式是否一致\n")
  cat("  影像 ID 样例: ", paste(head(img_df$ID, 5), collapse = ", "), "\n")
  cat("  临床 ID 样例: ", paste(head(clin_df$ID, 5), collapse = ", "), "\n")
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. 定义 17 个通路并验证                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

cat("\n[2] 通路验证...\n")

# 这是 run_extract_all.py 中的 HEADER 对应的 17 个通路列名
# 注意: CSV 中的原始列名 (check.names=FALSE 保留原名)
pathways_raw <- c("5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT",
                  "A4B2", "D1", "D2", "DAT", "M1", "NAT", "VAChT",
                  "human_CHA", "JHU_EC", "Lateral_Path", "Medial_Path")

# 检查哪些通路在合并后的数据中存在
available <- pathways_raw[pathways_raw %in% colnames(master_df)]
missing   <- pathways_raw[!pathways_raw %in% colnames(master_df)]

cat(sprintf("  找到 %d / %d 个通路列\n", length(available), length(pathways_raw)))
if (length(missing) > 0) {
  cat(sprintf("  ⚠️  缺失: %s\n", paste(missing, collapse = ", ")))
  # 如果是因为 check.names 导致的, 尝试修复
  alt_names <- paste0("X", missing)
  for (i in seq_along(missing)) {
    if (alt_names[i] %in% colnames(master_df)) {
      cat(sprintf("  → 找到替代名 '%s', 重命名为 '%s'\n", alt_names[i], missing[i]))
      colnames(master_df)[colnames(master_df) == alt_names[i]] <- missing[i]
      available <- c(available, missing[i])
    }
  }
}

pathways <- available
cat(sprintf("  最终分析通路 (%d): %s\n", length(pathways), paste(pathways, collapse = ", ")))

# TLV 列名检测
tlv_col <- if ("TLV" %in% colnames(master_df)) "TLV" else
           if ("TLV_mm3" %in% colnames(master_df)) "TLV_mm3" else
           stop("❌ 找不到总病灶体积列 (TLV 或 TLV_mm3)")
cat(sprintf("  总病灶体积列: '%s'\n", tlv_col))

# 确保数值型
master_df[[tlv_col]] <- as.numeric(master_df[[tlv_col]])
for (p in pathways) {
  master_df[[p]] <- as.numeric(master_df[[p]])
}

# TLV 基本统计
cat(sprintf("\n  TLV 统计:\n"))
cat(sprintf("    N (非缺失) = %d\n", sum(!is.na(master_df[[tlv_col]]))))
cat(sprintf("    Mean  = %.1f mm³\n", mean(master_df[[tlv_col]], na.rm = TRUE)))
cat(sprintf("    Median= %.1f mm³\n", median(master_df[[tlv_col]], na.rm = TRUE)))
cat(sprintf("    Range = [%.1f, %.1f]\n",
            min(master_df[[tlv_col]], na.rm = TRUE),
            max(master_df[[tlv_col]], na.rm = TRUE)))

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. 计算 Koch 损伤残差 (Residualization)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

cat("\n" %s+% strrep("=", 70) %s+% "\n")
cat("[3] 残差计算: Load_NT ~ TLV (Koch 2025, Fig.1E 方法学)\n")
cat(strrep("=", 70) %s+% "\n\n")

# 存放回归摘要
reg_summary <- tibble(
  Pathway        = character(),
  N_valid        = integer(),
  Beta_TLV       = double(),
  Intercept      = double(),
  R_squared      = double(),
  P_value        = double(),
  Std_Error      = double()
)

for (p in pathways) {
  resid_col <- paste0(p, "_resid")

  # 构建回归
  formula_obj <- as.formula(paste0("`", p, "` ~ `", tlv_col, "`"))

  fit <- lm(formula_obj, data = master_df, na.action = na.exclude)

  # 保存残差 (na.exclude 确保行数对齐)
  master_df[[resid_col]] <- residuals(fit)

  # 提取回归统计
  s <- summary(fit)
  n_valid <- sum(!is.na(master_df[[p]]) & !is.na(master_df[[tlv_col]]))

  reg_summary <- reg_summary %>% add_row(
    Pathway    = p,
    N_valid    = n_valid,
    Beta_TLV   = coef(fit)[2],
    Intercept  = coef(fit)[1],
    R_squared  = s$r.squared,
    P_value    = coef(s)[2, 4],
    Std_Error  = coef(s)[2, 2]
  )

  sig_mark <- ifelse(coef(s)[2, 4] < 0.001, "***",
              ifelse(coef(s)[2, 4] < 0.01, "**",
              ifelse(coef(s)[2, 4] < 0.05, "*", "")))

  cat(sprintf("  ✓ %s: β=%.4e, R²=%.4f, p=%.2e %s (N=%d)\n",
              p, coef(fit)[2], s$r.squared, coef(s)[2, 4], sig_mark, n_valid))
}

write.csv(reg_summary, file.path(OUT_DIR, "regression_summary.csv"), row.names = FALSE)
cat(sprintf("\n  📊 回归摘要已保存: regression_summary.csv\n"))

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. 智能识别临床指标列                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

cat("\n[4] 识别临床指标...\n")

# 智能搜索 IL-6 列 (可能叫 IL6, IL.6, IL-6, il6 等)
find_clinical_col <- function(df, patterns) {
  all_cols <- colnames(df)
  for (pat in patterns) {
    matched <- grep(pat, all_cols, ignore.case = TRUE, value = TRUE)
    if (length(matched) > 0) return(matched[1])
  }
  return(NA_character_)
}

# 定义我们感兴趣的临床指标及其可能的列名
clinical_targets <- list(
  IL6    = c("^IL6$", "^IL.6$", "^IL-6$", "^il6$", "interleukin.6", "IL_6"),
  IL10   = c("^IL10$", "^IL.10$", "^IL-10$", "interleukin.10"),
  CRP    = c("^CRP$", "^crp$", "^hsCRP$", "C.reactive"),
  TNFa   = c("^TNF", "^tnf", "tumor.necrosis"),
  NLR    = c("^NLR$", "neutrophil.lymphocyte"),
  WBC    = c("^WBC$", "^wbc$", "white.blood"),
  RMSSD  = c("RMSSD", "rmssd", "HRV_RMSSD"),
  SDNN   = c("SDNN", "sdnn", "HRV_SDNN"),
  HRn    = c("^HRn$", "^HR$", "heart.rate", "^hr$"),
  LF_HF  = c("LF.HF", "LF_HF", "lf.hf"),
  mRS    = c("^mRS$", "^mrs$", "mRS_90d", "modified.Rankin"),
  NIHSS  = c("^NIHSS$", "^nihss$"),
  Age    = c("^Age$", "^age$", "^AGE$"),
  Sex    = c("^Sex$", "^sex$", "^Gender$", "^gender$"),
  HAMD   = c("^HAMD$", "^hamd$", "Hamilton.Depression")
)

# 实际匹配
clinical_map <- sapply(names(clinical_targets), function(name) {
  find_clinical_col(master_df, clinical_targets[[name]])
})

cat("  临床变量映射:\n")
for (name in names(clinical_map)) {
  status <- if (is.na(clinical_map[name])) "❌ 未找到" else paste0("✓ → '", clinical_map[name], "'")
  cat(sprintf("    %-8s %s\n", name, status))
}

# 确保数值型
for (col_name in na.omit(clinical_map)) {
  if (col_name %in% colnames(master_df)) {
    master_df[[col_name]] <- suppressWarnings(as.numeric(master_df[[col_name]]))
  }
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. 寻找"冠军通路" — Correlation Fingerprint                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

cat("\n" %s+% strrep("=", 70) %s+% "\n")
cat("[5] Correlation Fingerprint: 递质残差 × 临床指标\n")
cat(strrep("=", 70) %s+% "\n")

resid_cols <- paste0(pathways, "_resid")

# --- 通用指纹分析函数 ---
compute_fingerprint <- function(df, resid_cols, clinical_col, method = "spearman",
                                covariates = NULL) {
  if (is.na(clinical_col) || !clinical_col %in% colnames(df)) {
    cat(sprintf("  ⚠️  临床变量 '%s' 不存在, 跳过\n", clinical_col))
    return(NULL)
  }

  results <- tibble(
    Pathway     = character(),
    Correlation = double(),
    P_Value     = double(),
    N           = integer(),
    Method      = character(),
    CI_Lower    = double(),
    CI_Upper    = double()
  )

  for (i in seq_along(resid_cols)) {
    rc <- resid_cols[i]
    pathway_name <- gsub("_resid$", "", rc)

    if (!rc %in% colnames(df)) next

    # 去除缺失值
    sub <- df[complete.cases(df[, c(rc, clinical_col)]), ]

    if (nrow(sub) < 10) {
      results <- results %>% add_row(
        Pathway = pathway_name, Correlation = NA, P_Value = NA,
        N = nrow(sub), Method = method, CI_Lower = NA, CI_Upper = NA
      )
      next
    }

    ct <- cor.test(sub[[rc]], sub[[clinical_col]], method = method,
                   conf.level = 0.95, exact = FALSE)

    results <- results %>% add_row(
      Pathway     = pathway_name,
      Correlation = ct$estimate,
      P_Value     = ct$p.value,
      N           = nrow(sub),
      Method      = method,
      CI_Lower    = if (!is.null(ct$conf.int)) ct$conf.int[1] else NA,
      CI_Upper    = if (!is.null(ct$conf.int)) ct$conf.int[2] else NA
    )
  }

  # FDR 校正 (Benjamini-Hochberg)
  if (nrow(results) > 0 && any(!is.na(results$P_Value))) {
    results$FDR_q <- p.adjust(results$P_Value, method = "BH")
  }

  # 排序: 按绝对相关系数降序
  results <- results %>% arrange(desc(abs(Correlation)))

  return(results)
}

# --- 5a. IL-6 指纹 ---
cat("\n  --- 5a. IL-6 (炎症) 指纹 ---\n")
il6_col <- clinical_map["IL6"]
fp_il6 <- compute_fingerprint(master_df, resid_cols, il6_col)

if (!is.null(fp_il6)) {
  cat("\n  🏆 IL-6 Neurotransmitter Fingerprint:\n")
  print(fp_il6 %>% select(Pathway, Correlation, P_Value, FDR_q, N), n = 17)
  write.csv(fp_il6, file.path(OUT_DIR, "fingerprint_IL6.csv"), row.names = FALSE)

  # 冠军通路
  top <- fp_il6 %>% slice(1)
  cat(sprintf("\n  🥇 冠军通路: %s (ρ = %.4f, p = %.2e, FDR q = %.4f)\n",
              top$Pathway, top$Correlation, top$P_Value,
              ifelse(is.na(top$FDR_q), NA, top$FDR_q)))
}

# --- 5b. RMSSD (迷走张力) 指纹 ---
cat("\n  --- 5b. RMSSD (迷走张力/自主神经) 指纹 ---\n")
rmssd_col <- clinical_map["RMSSD"]
fp_rmssd <- compute_fingerprint(master_df, resid_cols, rmssd_col)

if (!is.null(fp_rmssd)) {
  cat("\n  🏆 RMSSD Neurotransmitter Fingerprint:\n")
  print(fp_rmssd %>% select(Pathway, Correlation, P_Value, FDR_q, N), n = 17)
  write.csv(fp_rmssd, file.path(OUT_DIR, "fingerprint_HRV.csv"), row.names = FALSE)
}

# --- 5c. HRn 指纹 ---
cat("\n  --- 5c. HRn (心率) 指纹 ---\n")
hrn_col <- clinical_map["HRn"]
fp_hrn <- compute_fingerprint(master_df, resid_cols, hrn_col)

if (!is.null(fp_hrn)) {
  cat("\n  🏆 HRn Neurotransmitter Fingerprint:\n")
  print(fp_hrn %>% select(Pathway, Correlation, P_Value, FDR_q, N), n = 17)
}

# --- 5d. 全面扫描: 所有临床变量 ---
cat("\n  --- 5d. 全面扫描 ---\n")

all_fingerprints <- tibble()
for (clin_name in names(clinical_map)) {
  col <- clinical_map[clin_name]
  if (is.na(col)) next

  fp <- compute_fingerprint(master_df, resid_cols, col)
  if (!is.null(fp)) {
    fp$Clinical_Variable <- clin_name
    fp$Clinical_Column   <- col
    all_fingerprints <- bind_rows(all_fingerprints, fp)
  }
}

if (nrow(all_fingerprints) > 0) {
  write.csv(all_fingerprints, file.path(OUT_DIR, "fingerprint_all_clinical.csv"),
            row.names = FALSE)
  cat(sprintf("  📊 全部指纹已保存 (%d 组合)\n", nrow(all_fingerprints)))

  # 打印最显著的 Top 20
  cat("\n  🔥 Top 20 最强关联 (跨所有指标):\n")
  top20 <- all_fingerprints %>%
    filter(!is.na(P_Value)) %>%
    arrange(P_Value) %>%
    head(20)
  print(top20 %>% select(Clinical_Variable, Pathway, Correlation, P_Value, FDR_q, N))
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  6. 偏相关分析 (控制年龄、性别、NIHSS)                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

cat("\n" %s+% strrep("=", 70) %s+% "\n")
cat("[6] 偏相关分析 (控制协变量后的 \"纯\" 关联)\n")
cat(strrep("=", 70) %s+% "\n")

# 偏相关函数 (使用线性回归残差法)
partial_cor <- function(x, y, z_mat) {
  # x, y: 目标变量; z_mat: 协变量矩阵
  complete <- complete.cases(cbind(x, y, z_mat))
  if (sum(complete) < 20) return(list(r = NA, p = NA, n = sum(complete)))

  x <- x[complete]; y <- y[complete]; z_mat <- z_mat[complete, , drop = FALSE]

  # 从 x 和 y 中各自回归掉协变量的影响
  resid_x <- residuals(lm(x ~ ., data = as.data.frame(z_mat)))
  resid_y <- residuals(lm(y ~ ., data = as.data.frame(z_mat)))

  ct <- cor.test(resid_x, resid_y, method = "spearman", exact = FALSE)
  return(list(r = ct$estimate, p = ct$p.value, n = sum(complete)))
}

# 构建协变量矩阵
covariates_cols <- c()
for (cov_name in c("Age", "Sex", "NIHSS")) {
  col <- clinical_map[cov_name]
  if (!is.na(col) && col %in% colnames(master_df)) {
    covariates_cols <- c(covariates_cols, col)
  }
}

if (length(covariates_cols) >= 1) {
  cat(sprintf("  控制变量: %s\n\n", paste(covariates_cols, collapse = ", ")))

  z_mat <- master_df[, covariates_cols, drop = FALSE]
  for (j in seq_along(covariates_cols)) {
    z_mat[[j]] <- as.numeric(z_mat[[j]])
  }

  # 对 IL-6 做偏相关
  if (!is.na(il6_col)) {
    cat("  --- IL-6 偏相关 (控制 Age, Sex, NIHSS) ---\n")
    pcor_il6 <- tibble(Pathway = character(), PartialCor = double(),
                       P_Value = double(), N = integer())

    for (rc in resid_cols) {
      pname <- gsub("_resid$", "", rc)
      if (!rc %in% colnames(master_df)) next

      pc <- partial_cor(master_df[[rc]], master_df[[il6_col]], z_mat)
      pcor_il6 <- pcor_il6 %>% add_row(
        Pathway = pname, PartialCor = pc$r, P_Value = pc$p, N = pc$n
      )
    }

    pcor_il6$FDR_q <- p.adjust(pcor_il6$P_Value, method = "BH")
    pcor_il6 <- pcor_il6 %>% arrange(desc(abs(PartialCor)))

    cat("\n  🏆 IL-6 偏相关排名:\n")
    print(pcor_il6, n = 17)
    write.csv(pcor_il6, file.path(OUT_DIR, "partial_cor_IL6.csv"), row.names = FALSE)
  }

  # 对 RMSSD 做偏相关
  if (!is.na(rmssd_col)) {
    cat("\n  --- RMSSD 偏相关 (控制 Age, Sex, NIHSS) ---\n")
    pcor_rmssd <- tibble(Pathway = character(), PartialCor = double(),
                         P_Value = double(), N = integer())

    for (rc in resid_cols) {
      pname <- gsub("_resid$", "", rc)
      if (!rc %in% colnames(master_df)) next

      pc <- partial_cor(master_df[[rc]], master_df[[rmssd_col]], z_mat)
      pcor_rmssd <- pcor_rmssd %>% add_row(
        Pathway = pname, PartialCor = pc$r, P_Value = pc$p, N = pc$n
      )
    }

    pcor_rmssd$FDR_q <- p.adjust(pcor_rmssd$P_Value, method = "BH")
    pcor_rmssd <- pcor_rmssd %>% arrange(desc(abs(PartialCor)))

    cat("\n  🏆 RMSSD 偏相关排名:\n")
    print(pcor_rmssd, n = 17)
    write.csv(pcor_rmssd, file.path(OUT_DIR, "partial_cor_RMSSD.csv"), row.names = FALSE)
  }
} else {
  cat("  ⚠️  未找到可用的协变量 (Age/Sex/NIHSS), 跳过偏相关\n")
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  7. 可视化 — 递质损伤指纹图谱 (Koch 2025, Fig.2 style)                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

cat("\n" %s+% strrep("=", 70) %s+% "\n")
cat("[7] 绘制指纹图谱\n")
cat(strrep("=", 70) %s+% "\n")

# Koch/Brain 论文配色
nt_colors <- c(
  "5HT1a" = "#4DBBD5", "5HT1b" = "#4DBBD5", "5HT2a" = "#7DCDE5",
  "5HT4"  = "#A8DFF0", "5HT6"  = "#B0E0F6", "5HTT"  = "#3B9FC4",
  "A4B2"  = "#F39B7F",
  "D1"    = "#E64B35", "D2"    = "#DC7C6B", "DAT"   = "#E64B35",
  "M1"    = "#8491B4",
  "NAT"   = "#91D1C2",
  "VAChT" = "#00A087",
  "human_CHA"    = "#2E8B57",
  "JHU_EC"       = "#3CB371",
  "Lateral_Path" = "#228B22",
  "Medial_Path"  = "#006400"
)

# 递质系统分组标签
nt_system <- c(
  "5HT1a" = "Serotonin", "5HT1b" = "Serotonin", "5HT2a" = "Serotonin",
  "5HT4"  = "Serotonin", "5HT6"  = "Serotonin", "5HTT"  = "Serotonin",
  "A4B2"  = "Nicotinic",
  "D1"    = "Dopamine", "D2"    = "Dopamine", "DAT"   = "Dopamine",
  "M1"    = "Muscarinic",
  "NAT"   = "Noradrenaline",
  "VAChT" = "Cholinergic",
  "human_CHA"    = "Cholinergic Pathway",
  "JHU_EC"       = "Cholinergic Pathway",
  "Lateral_Path" = "Cholinergic Pathway",
  "Medial_Path"  = "Cholinergic Pathway"
)

# --- 7a. IL-6 指纹柱状图 ---
plot_fingerprint <- function(fp_data, title_text, outcome_label, filename_base) {
  if (is.null(fp_data) || nrow(fp_data) == 0) return(invisible(NULL))

  # 添加系统分组和颜色
  fp_data <- fp_data %>%
    mutate(
      System     = nt_system[Pathway],
      Color      = nt_colors[Pathway],
      Significant = ifelse(!is.na(FDR_q) & FDR_q < 0.05, "FDR < 0.05",
                    ifelse(!is.na(P_Value) & P_Value < 0.05, "p < 0.05", "n.s.")),
      Pathway_label = factor(Pathway, levels = Pathway)  # 保持排序
    )

  # 主图
  p <- ggplot(fp_data, aes(x = reorder(Pathway, Correlation),
                            y = Correlation, fill = Significant)) +
    geom_bar(stat = "identity", width = 0.75) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.3) +
    coord_flip() +
    scale_fill_manual(
      values = c("FDR < 0.05" = "#C62828", "p < 0.05" = "#EF5350", "n.s." = "gray70"),
      name = "Significance"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor   = element_blank(),
      legend.position     = c(0.85, 0.15),
      legend.background   = element_rect(fill = alpha("white", 0.8), color = NA),
      plot.title          = element_text(face = "bold", size = 13),
      plot.subtitle       = element_text(size = 10, color = "gray40"),
      axis.text.y         = element_text(size = 10)
    ) +
    labs(
      title    = title_text,
      subtitle = sprintf("Koch (2025) Residualized Damage | N ≈ %d | Spearman",
                         fp_data$N[1]),
      x        = "Neurotransmitter / Pathway (Specific Damage)",
      y        = sprintf("Correlation with %s (ρ)", outcome_label)
    )

  # 添加数值标签
  p <- p + geom_text(aes(label = sprintf("%.3f", Correlation)),
                     hjust = ifelse(fp_data$Correlation >= 0, -0.1, 1.1),
                     size = 3, color = "gray30")

  # 保存
  ggsave(file.path(fig_dir, paste0(filename_base, ".pdf")), p,
         width = 10, height = 7)
  ggsave(file.path(fig_dir, paste0(filename_base, ".png")), p,
         width = 10, height = 7, dpi = 300)

  cat(sprintf("  ✓ %s 已保存\n", filename_base))
  return(p)
}

p_il6 <- plot_fingerprint(
  fp_il6,
  "Neurotransmitter Fingerprint of Post-stroke Inflammation (IL-6)",
  "IL-6",
  "fig_fingerprint_IL6"
)

p_rmssd <- plot_fingerprint(
  fp_rmssd,
  "Neurotransmitter Fingerprint of Cardiac Autonomic Function (RMSSD)",
  "RMSSD",
  "fig_fingerprint_RMSSD"
)

p_hrn <- plot_fingerprint(
  fp_hrn,
  "Neurotransmitter Fingerprint of Heart Rate (HRn)",
  "HRn",
  "fig_fingerprint_HRn"
)

# --- 7b. Load vs TLV 散点图 (Koch Fig.1E) ---
cat("\n  绘制 Load ~ TLV 散点图...\n")

n_pw <- length(pathways)
ncols <- 4
nrows <- ceiling(n_pw / ncols)

pdf(file.path(fig_dir, "fig_load_vs_tlv.pdf"), width = 5 * ncols, height = 4 * nrows)
par(mfrow = c(nrows, ncols), mar = c(4, 4, 3, 1))

for (p in pathways) {
  x <- master_df[[tlv_col]]
  y <- master_df[[p]]
  valid <- !is.na(x) & !is.na(y)

  if (sum(valid) < 10) {
    plot.new(); title(paste(p, "- insufficient data"))
    next
  }

  fit <- lm(y[valid] ~ x[valid])
  resid_vals <- residuals(fit)
  r2 <- summary(fit)$r.squared
  pval <- coef(summary(fit))[2, 4]

  col_vec <- ifelse(resid_vals >= 0,
                    ifelse(p %in% names(nt_colors), nt_colors[p], "#E64B35"),
                    "lightgray")

  plot(x[valid], y[valid], pch = 16, cex = 0.3, col = alpha(col_vec, 0.4),
       xlab = "TLV (mm³)", ylab = "Weighted Load",
       main = sprintf("%s\nR²=%.3f, p=%.1e", p, r2, pval),
       cex.main = 0.9)
  abline(fit, col = "black", lwd = 2, lty = 2)
}

dev.off()

png(file.path(fig_dir, "fig_load_vs_tlv.png"), width = 5 * ncols * 100,
    height = 4 * nrows * 100, res = 150)
par(mfrow = c(nrows, ncols), mar = c(4, 4, 3, 1))
for (p in pathways) {
  x <- master_df[[tlv_col]]
  y <- master_df[[p]]
  valid <- !is.na(x) & !is.na(y)
  if (sum(valid) < 10) { plot.new(); title(paste(p, "- insufficient data")); next }
  fit <- lm(y[valid] ~ x[valid])
  resid_vals <- residuals(fit)
  r2 <- summary(fit)$r.squared
  pval <- coef(summary(fit))[2, 4]
  col_vec <- ifelse(resid_vals >= 0,
                    ifelse(p %in% names(nt_colors), nt_colors[p], "#E64B35"),
                    "lightgray")
  plot(x[valid], y[valid], pch = 16, cex = 0.3, col = alpha(col_vec, 0.4),
       xlab = "TLV (mm³)", ylab = "Weighted Load",
       main = sprintf("%s\nR²=%.3f, p=%.1e", p, r2, pval), cex.main = 0.9)
  abline(fit, col = "black", lwd = 2, lty = 2)
}
dev.off()
cat("  ✓ fig_load_vs_tlv 已保存\n")

# --- 7c. 残差 × 临床相关矩阵热图 ---
cat("\n  绘制相关矩阵热图...\n")

clinical_available <- na.omit(clinical_map)
if (length(clinical_available) >= 2) {
  # 构建矩阵: 行 = 通路残差, 列 = 临床指标
  cor_mat <- matrix(NA, nrow = length(resid_cols), ncol = length(clinical_available),
                    dimnames = list(gsub("_resid$", "", resid_cols),
                                   names(clinical_available)))
  pval_mat <- cor_mat

  for (i in seq_along(resid_cols)) {
    for (j in seq_along(clinical_available)) {
      rc <- resid_cols[i]
      cc <- clinical_available[j]
      if (!rc %in% colnames(master_df) || !cc %in% colnames(master_df)) next
      sub <- master_df[complete.cases(master_df[, c(rc, cc)]), ]
      if (nrow(sub) < 10) next
      ct <- cor.test(sub[[rc]], as.numeric(sub[[cc]]), method = "spearman", exact = FALSE)
      cor_mat[i, j]  <- ct$estimate
      pval_mat[i, j] <- ct$p.value
    }
  }

  # 只保留至少有一个非 NA 的行/列
  valid_rows <- apply(cor_mat, 1, function(x) any(!is.na(x)))
  valid_cols <- apply(cor_mat, 2, function(x) any(!is.na(x)))

  if (sum(valid_rows) >= 2 && sum(valid_cols) >= 2) {
    cor_sub  <- cor_mat[valid_rows, valid_cols, drop = FALSE]
    pval_sub <- pval_mat[valid_rows, valid_cols, drop = FALSE]

    pdf(file.path(fig_dir, "fig_heatmap.pdf"), width = max(8, ncol(cor_sub) * 0.8 + 3),
        height = max(6, nrow(cor_sub) * 0.5 + 2))
    corrplot(cor_sub, method = "color", type = "full",
             col = colorRampPalette(c("#2166AC", "white", "#B2182B"))(200),
             tl.col = "black", tl.srt = 45, tl.cex = 0.8,
             cl.cex = 0.7,
             p.mat = pval_sub, sig.level = 0.05, insig = "label_sig",
             pch.cex = 0.8,
             title = "Residualized NT Damage × Clinical Outcomes\n(Spearman, * p<0.05)",
             mar = c(0, 0, 3, 0))
    dev.off()

    png(file.path(fig_dir, "fig_heatmap.png"), width = max(800, ncol(cor_sub) * 80 + 300),
        height = max(600, nrow(cor_sub) * 50 + 200), res = 150)
    corrplot(cor_sub, method = "color", type = "full",
             col = colorRampPalette(c("#2166AC", "white", "#B2182B"))(200),
             tl.col = "black", tl.srt = 45, tl.cex = 0.8, cl.cex = 0.7,
             p.mat = pval_sub, sig.level = 0.05, insig = "label_sig",
             pch.cex = 0.8,
             title = "Residualized NT Damage × Clinical Outcomes\n(Spearman, * p<0.05)",
             mar = c(0, 0, 3, 0))
    dev.off()
    cat("  ✓ fig_heatmap 已保存\n")
  }
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  8. 保存完整数据 & 最终报告                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

cat("\n" %s+% strrep("=", 70) %s+% "\n")
cat("[8] 保存数据与总结\n")
cat(strrep("=", 70) %s+% "\n")

# 保存合并 + 残差后的完整数据
write.csv(master_df, file.path(OUT_DIR, "master_merged.csv"), row.names = FALSE)
cat(sprintf("  📦 完整数据已保存: master_merged.csv (%d 行 × %d 列)\n",
            nrow(master_df), ncol(master_df)))

# 只保存残差列
resid_export <- master_df %>% select(ID, all_of(tlv_col), all_of(resid_cols))
write.csv(resid_export, file.path(OUT_DIR, "residuals_table.csv"), row.names = FALSE)
cat(sprintf("  📦 残差表已保存: residuals_table.csv\n"))

# --- 最终摘要 ---
cat("\n" %s+% strrep("═", 70) %s+% "\n")
cat("  🎯 分析完成! 关键结果摘要\n")
cat(strrep("═", 70) %s+% "\n\n")

cat(sprintf("  样本量:         %d\n", nrow(master_df)))
cat(sprintf("  分析通路数:     %d\n", length(pathways)))
cat(sprintf("  输出目录:       %s\n", OUT_DIR))
cat(sprintf("  图形目录:       %s\n", fig_dir))

if (!is.null(fp_il6) && nrow(fp_il6) > 0) {
  top1 <- fp_il6 %>% slice(1)
  cat(sprintf("\n  🏆 IL-6 冠军通路: %s (ρ=%.4f, p=%.2e)\n",
              top1$Pathway, top1$Correlation, top1$P_Value))

  # 胆碱能通路排名
  cho <- fp_il6 %>%
    filter(Pathway %in% c("VAChT", "human_CHA", "JHU_EC", "Lateral_Path", "Medial_Path"))
  if (nrow(cho) > 0) {
    cat("\n  胆碱能通路 IL-6 排名:\n")
    for (i in 1:nrow(cho)) {
      sig <- ifelse(!is.na(cho$FDR_q[i]) & cho$FDR_q[i] < 0.05, "✓ FDR显著",
             ifelse(!is.na(cho$P_Value[i]) & cho$P_Value[i] < 0.05, "* p显著", ""))
      cat(sprintf("    %-15s ρ = %+.4f  p = %.2e  %s\n",
                  cho$Pathway[i], cho$Correlation[i], cho$P_Value[i], sig))
    }
  }
}

if (!is.null(fp_rmssd) && nrow(fp_rmssd) > 0) {
  top1 <- fp_rmssd %>% slice(1)
  cat(sprintf("\n  🏆 RMSSD 冠军通路: %s (ρ=%.4f, p=%.2e)\n",
              top1$Pathway, top1$Correlation, top1$P_Value))
}

cat("\n" %s+% strrep("═", 70) %s+% "\n")
cat("  下一步: 中介效应分析 (Mediation)\n")
cat("  病灶 → 胆碱能断连(残差) → 炎症(IL-6) → 预后(mRS)\n")
cat("  请运行 koch_mediation.R (即将创建)\n")
cat(strrep("═", 70) %s+% "\n")
