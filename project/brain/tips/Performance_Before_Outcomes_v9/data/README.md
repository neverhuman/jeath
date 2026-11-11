# Third-party data

Redistributed unmodified, with attribution, under the licenses stated by their sources.

| File | Source | License |
|---|---|---|
| `gingerich2022/41598_2022_15481_MOESM1_ESM.xlsx` | Gingerich, P. D. (2022). Pattern and rate in the Plio-Pleistocene evolution of modern human brain size. *Sci. Rep.* 12:11216. doi:10.1038/s41598-022-15481-3. Supplementary Dataset 1 (233 specimens). | CC BY 4.0 |
| `will2021/41467_2021_24290_MOESM5_ESM.xlsx` | Will, M., Krapp, M., Stock, J. T., Manica, A. (2021). Different environmental variables predict body and brain size evolution in *Homo*. *Nat. Commun.* 12:4116. doi:10.1038/s41467-021-24290-7. Supplementary File 2 (166 specimens). | CC BY 4.0 |
| `du2018/du2018_ecv.xlsx` | Du, A. et al. (2018). Data from: Pattern and process in hominin brain size evolution are scale-dependent. Dryad, doi:10.5061/dryad.c30g9 (94 specimens). | CC0 1.0 |

The loaders are in `code/load_fossils.py`. Regional assignment of Middle Pleistocene specimens (used for Fig. 2B) is derived
from the site coordinates in the Will et al. file and is coded there.

## General Social Survey (not redistributed)

`code/fetch_gss.py` clones the public R data package `kjhealy/gssr` (MIT-licensed packaging of the NORC General Social
Survey cumulative file, 1972-2024) into `data/gss/gssr/`, and writes the 36 variables used by the analysis to
`data/gss/gss_extract.npz`. The survey data are (c) NORC at the University of Chicago and are free to use with citation:
Davern, M., Bautista, R., Freese, J., Herd, P., and Morgan, S. L., General Social Survey 1972-2024, NORC, Chicago.
The analysis reports aggregates by sex and birth cohort only.

## Columbia speed-dating experiment (not redistributed)

`code/fetch_speed.py` copies the public file of the Columbia speed-dating experiment (Fisman, R., Iyengar, S. S., Kamenica,
E., and Simonson, I., 2006, *Q. J. Econ.* 121:673-697; 8,378 rows, one per person per meeting) from a public mirror to
`data/speeddating/speed_dating.csv` and checks its SHA-256 digest
(`fda9a924b18880fabe6c7a71961cff20aa3c7cc2a5aa8d6e823c2fc41893814c`); four mirrors carried the identical file when this
version was built. The analysis (`code/exp_speed.py`) reads ratings, decisions, stated preferences, sex, age and session
identifiers. It does not read the variables that describe race or religion, and it reports aggregates only.

## Ethnographic codes (not redistributed)

`code/fetch_dplace.py` makes a sparse clone of the public D-PLACE data repository (Kirby et al. 2016, *PLoS ONE* 11:e0158391;
commit `9bfed2c8c206be00f55f71516f262bbca2234e5a`, 7 February 2022) into `data/dplace/mirror` and keeps the Ethnographic
Atlas, the Standard Cross-Cultural Sample and Binford's hunter-gatherer compilation. `code/exp_cultures.py` tabulates
published codes on who chooses a spouse, consent, rape, captives, wife-beating, divorce, ages at marriage, polygyny, games
and marriage transfers. Societies are units of replication; none is ranked.

## Cross-national surveys (not redistributed)

`code/fetch_open.py` downloads, and checks by SHA-256 digest, (1) the data set of Kowal et al. 2025 (*Scientific Data*;
117,293 respondents in 175 countries; OSF project k54fx, licensed CC BY 4.0) and (2) the processed data of Walter et al.
2020 (*Psychological Science* 31:408-423; 14,399 respondents in 45 countries; OSF project gb5cn). `code/exp_world.py` reads
sex, age, country, relationship status, partner's age, self-ratings, ideal-partner ratings, the budget task, time on
social media and an attitude scale. It does not read the variables on religion, political views or sexual orientation.
No country is named or ranked in the paper.

## Second survey extract

`code/fetch_gss.py` also writes `data/gss/gss_extract2.npz` (marital status, children, ideal number of children, social
evenings, worship, hours online, partners in the past year), used by `code/exp_village.py`. The first extract is unchanged.
