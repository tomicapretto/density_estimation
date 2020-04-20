library(readr)
library(dplyr)

output_files <- paste0("../output/", list.files("../output"))
data_all <- bind_rows(lapply(output_files, read_csv))

saveRDS(data_all, "app/data/data.rds")
