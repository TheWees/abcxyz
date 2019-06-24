
start_all <- proc.time()

# devtools::install_github("briatte/ggnet")
# devtools::install_github("thomasp85/ggraph#78", dependencies=TRUE)

library(tidyverse, quietly = TRUE, warn.conflicts = FALSE) # for data wrangling
library(igraph, quietly = TRUE, warn.conflicts = FALSE)    # for network data structures and tools
# library(ggnet, quietly = TRUE, warn.conflicts = FALSE)       # for network plotting
# library(ggraph, quietly = TRUE, warn.conflicts = FALSE)       # for network plotting
library(lubridate, quietly = TRUE, warn.conflicts = FALSE) # for date manipulating
library(scales, quietly = TRUE, warn.conflicts = FALSE)
library(tm, quietly = TRUE, warn.conflicts = FALSE)
library(tidytext, quietly = TRUE, warn.conflicts = FALSE)
library(lsa, quietly = TRUE, warn.conflicts = FALSE)
library(quanteda, quietly = TRUE, warn.conflicts = FALSE)

# Load in data
investments <- read.csv("./Dataset/investments.csv", na.strings = c("", "NA"))
companies <- read.csv("./Dataset/companies.csv", na.strings = c("", "NA"))

# configurations
has_SEA = T
has_CHN = F
# has_USA = F
start_date = 2010
end_date = 2015

# # cos in earlier dates the startup which is not an investor yet can appear as an investor in the later years
# min_start_date = 2010
# max_end_date = 2015

# however as we are now doing link prediction rather than just plotting the communities to see, 
# we are not supposed to know the future, so we need not special configuration for min_start_date and max_end_date
# if the node appears in the other side of the graph, it will be treated as a new node
min_start_date = start_date
max_end_date = end_date

# setup
file_initials = ""
SEA = c("BRN", "KHM", "TLS", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM")
CHN = c("CHN", "HKG")
# USA = c("USA")
countries = c()
if(has_SEA) {
  countries = SEA
  file_initials = paste0(file_initials, "SEA_")
} 
if(has_CHN) {
  countries = append(countries, CHN)
  file_initials = paste0(file_initials, "CHN_")
} 
# if(has_USA){
#     countries = append(countries, USA)
#     file_initials = paste0(file_initials, "USA_")
# }
countries
file_initials = paste0(file_initials, start_date, "-", end_date)
file_initials

# Clean data - step 1 - select only required fields, label seq of records

ptm <- proc.time()
investments_01 <- investments %>% 
  dplyr::select(company_permalink, company_name, company_category_list,
                company_country_code, company_region, 
                investor_permalink, investor_name, 
                investor_country_code, investor_region,
                funding_round_type, funded_at, raised_amount_usd) %>% 
  group_by(company_permalink) %>%
  mutate(
    funding_seq = dense_rank(funded_at),
    funded_at = as.Date(funded_at, '%d/%m/%y'),
    funded_at_ts = as.numeric(as.POSIXct(funded_at, '%d/%m/%y'))
  ) %>%
  ungroup %>% unique
proc.time() - ptm

investments_01

# Clean data - step 2 - Remove all the "organization" in startup companies, 
# Rename those in investor and company for bipartite graph to use
# suffix with "_i" if it is an investor, suffix with "_s" if it is a startup
investments_01a <- investments_01
# investments_02

investments_01a$organization_or_person <- investments_01$investor_permalink %>% 
  gsub("(\\/?)(.*)(\\/.*)", "\\2", .) 

investments_01a$company_permalink <- investments_01$company_permalink %>% 
  gsub("(\\/?)(.*)(\\/)(.*)", "\\4", .) 
# investments_02
investments_01a$investor_permalink <- investments_01$investor_permalink %>% 
  gsub("(\\/?)(.*)(\\/)(.*)", "\\4", .) 
investments_01a

investments_02 <- investments_01a

investments_02 <- investments_02 %>% 
  filter(funded_at >= as.Date(paste0(min_start_date, "-01-01"))) %>%
  filter(funded_at < as.Date(paste0(max_end_date+1, "-01-01"))) %>%
  mutate(investor_permalink = ifelse(investor_permalink %in% investments_01a$company_permalink, 
                                     paste0(investor_permalink, "_i"), investor_permalink),
         company_permalink = ifelse(company_permalink %in% investments_01a$investor_permalink, 
                                    paste0(company_permalink, "_s"), company_permalink))
investments_02

# See number of empty records

investments_02 %>%
  mutate_all(is.na) %>%
  gather(key = "variable_name", value="is_na_count") %>% 
  group_by(variable_name) %>% 
  summarise(
    is_na_count=sum(is_na_count), 
    percentage = sum(is_na_count) / nrow(investments_01) * 100) %>%
  arrange(desc(is_na_count))

unique(investments_02$company_country_code)

# Clean data - step 3 - remove empty rows and further subset by region and date

data_subset <- investments_02[ , c("company_permalink", "investor_permalink", "raised_amount_usd", 
                                   "company_region", "company_category_list")] 
investments_03 <- investments_02[complete.cases(data_subset), ] # Omit NAs by columns
investments_03

investments_03 <- investments_03 %>% 
  filter(company_country_code %in% countries) %>%
  filter(funded_at >= as.Date(paste0(start_date, "-01-01"))) %>%
  filter(funded_at < as.Date(paste0(end_date+1, "-01-01"))) %>%
  arrange(funded_at, funding_seq) %>%
  filter(company_name != ".") #there is a strange company not in the company.csv

dim(investments_03)
investments_03

length(unique(investments_03$company_permalink))

# Clean data - step 4 - categories substring, remove crowdfunding
investments_04 <- investments_03

investments_04$company_category_list <- investments_03$company_category_list %>% 
  gsub("(.*?)(\\|.*)", "\\1", .)

# investments_03$funding_round_type %>% unique
# investments_03 %>% filter(grepl("crowdfunding", .$funding_round_type)) %>% count() #254 rows
investments_04 <- investments_04 %>% 
  rename(company_category = company_category_list) %>% 
  filter(!grepl("crowdfunding", .$funding_round_type)) %>% droplevels

dim(investments_04)
# investments_03 %>% group_by(funding_round_type) %>% summarise(count=n()) %>% arrange(desc(count))
head(investments_04 %>% group_by(company_category) %>% summarise(count=n()) %>% arrange(desc(count)))


# Plot differences in investment amount over the 4 periods

# investments_periods <- investments_04 %>%
#   mutate(
#     funded_at_month = as.Date(cut(funded_at, breaks = "month")),
#     funded_at_year = as.Date(cut(funded_at, breaks = "month"))
#   ) %>%
#   group_by(funded_at_month) %>%
#   summarise(mean_funding=mean(raised_amount_usd))

# investments_periods
# ggplot(data=investments_periods) + geom_line(aes(x=funded_at_month, y=mean_funding)) +
#   scale_x_date(labels = date_format("%b-%y"),breaks = date_breaks("years"))


# Clean data - step 5 - Remove all the "organization" in startup companies
# Select out companies in the filtered set

investments_companies <- investments_04 %>% distinct(company_permalink, company_name, .keep_all = TRUE)
dim(investments_companies)
investments_companies

companies_01 <- companies
companies_01$category_list <- companies$category_list %>% 
  gsub("(.*?)(\\|.*)", "\\1", .)

companies_01$permalink <- companies$permalink %>% 
  gsub("(\\/?)(.*)(\\/)(.*)", "\\4", .) 
companies_01

companies_01 <- companies_01 %>% 
  filter(permalink %in% investments_companies$company_permalink) %>%
  rename(category = category_list) %>% 
  select(permalink, name, category, country_code, region, status)

any(is.na(companies_01$status)) #FALSE - none of the status is NA

dim(companies_01)
companies_01

# Finalise investors, investment & company data to csv

investments_final <-investments_04
write.csv(investments_final, paste0("./DatasetProcessed_v9/investments_final_", file_initials, ".csv"), row.names = FALSE)
companies_final <- companies_01
write.csv(companies_final, paste0("./DatasetProcessed_v9/companies_final_", file_initials, ".csv"), row.names = FALSE)

investor_nodes <- investments_final %>% select(investor_permalink) %>% 
  rename(uid = investor_permalink)
nrow(investor_nodes)
company_nodes <- companies_final %>% select(permalink) %>% 
  rename(uid = permalink)
nrow(company_nodes)

# Aggregating the count of investors / amount of money invested

investments_final_agg <- investments_final %>% 
  group_by(company_permalink, company_country_code, company_region) %>%
  mutate(company_funding_to_date = cumsum(raised_amount_usd)) %>%
  ungroup %>%
  group_by(investor_permalink) %>%
  mutate(investor_funding_to_date = cumsum(raised_amount_usd)) %>%
  ungroup %>%
  rename(investor_type=organization_or_person,
         to=company_permalink,
         from=investor_permalink) %>%
  select(to, from, company_country_code, company_region, company_category, 
         company_funding_to_date, investor_type, investor_funding_to_date,
         funded_at_ts, raised_amount_usd) %>% 
  ungroup %>% unique
nrow(investments_final_agg %>% unique)
investments_final_agg

if(has_SEA) { # use country
  investments_final_agg <- investments_final_agg %>% select(-company_region) %>%
    rename(company_origin=company_country_code)
} else { # use region
  investments_final_agg <- investments_final_agg %>% select(-company_country_code) %>%
    rename(company_origin=company_region)
}

# Handle the large number of categories by grouping them
category_grp_df <- read.csv("category_groups.csv", colClasses = c("character", "character"))
head(category_grp_df)

trim <- function (x) gsub("^\\s+|\\s+$", "", x)
category_grp_list <- list()
for (row in 1:nrow(category_grp_df)) {
  vec <- unlist(lapply(strsplit(category_grp_df[row, "Categories"], ","), trim))
  for (cat in vec) {
    category_grp_list[[cat]] <- category_grp_df[row, "Category.Group"]
  }
}
category_grp_list <- enframe(unlist(category_grp_list)) %>% rename(company_category=name, company_category_grp=value)

# update category grp
investments_final_agg <- investments_final_agg %>% merge(category_grp_list, by="company_category", all.x = TRUE) %>% select(-company_category)
investments_final_agg$company_category_grp[is.na(investments_final_agg$company_category_grp)] <- "Others"

write.csv(investments_final_agg, file=paste0("NodesNEdges/", 
                                             file_initials, "_edges_for_link_pred.csv"), row.names = F)


proc.time() - start_all


